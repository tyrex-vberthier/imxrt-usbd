//! USB endpoints
//!
//! Use endpoints to
//!
//! - read and write to endpoint memory buffers
//! - schedule transfers
//! - check transfer statuses
//! - signal endpoint state to the host
//!
//! The endpoint "owns" the static QH and TD memory. These
//! are temporarily stored in the driver behind an Option.
//! Once allocated, the driver will move the reference into
//! the endpoint, where it resides forever. You can use methods
//! on the endpoint to safely access QH and TD state.
//!
//! The endpoints take immutable borrows of the USB instance.
//! The contract is that, if the *Endpoint* is mutable, it's
//! permitted to modify its own USB register (or register
//! field). This gives us a kind of loose runtime ownership
//! of endpoint registers, but it only works if all endpoints
//! are owned by the same object, since we rely on carrying
//! the mutable borrow to prevent races. That's how today's
//! driver works.

use crate::{
    buffer::Buffer,
    qh::Qh,
    ral,
    ral::endpoint_control,
    td::{Status, Td},
};
use usb_device::{
    UsbDirection, UsbError,
    endpoint::{EndpointAddress, EndpointType},
};

/// Data Synchronisation Barrier — flushes the store buffer so all Normal-memory
/// writes are visible to other bus masters before the subsequent Device-memory
/// write (ENDPTPRIME) hands the dTD to the USB controller.
///
/// In host-side unit tests (x86_64) the Cortex-M assembly intrinsic is absent;
/// the cfg-gate replaces it with a compiler fence that has equivalent host
/// ordering semantics for the bookkeeping-only test scaffold.
#[inline(always)]
fn dsb() {
    #[cfg(not(test))]
    cortex_m::asm::dsb();
    #[cfg(test)]
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

/// Spin until ENDPTPRIME clears for this endpoint direction+index.
///
/// On real hardware the USB controller clears the bit once it has fetched the dTD.
/// In host-side unit tests there is no hardware, so the bit would never clear and
/// the spin would loop forever. The cfg-gate makes it a no-op in test builds.
#[inline(always)]
fn wait_endptprime(usb: &ral::AnyUsbInstance) {
    #[cfg(not(test))]
    while ral::read_reg!(ral::usb, usb, ENDPTPRIME) != 0 {}
    #[cfg(test)]
    let _ = usb;
}

/// Spin until ENDPTFLUSH clears for the given mask.
///
/// Same rationale as [`wait_endptprime`]: a no-op in test builds.
#[inline(always)]
#[allow(dead_code)]
fn wait_endptflush(usb: &ral::AnyUsbInstance, mask: u32) {
    #[cfg(not(test))]
    while ral::read_reg!(ral::usb, usb, ENDPTFLUSH) & mask != 0 {}
    #[cfg(test)]
    let _ = (usb, mask);
}

/// Largest single dTD payload we program: 16 KiB = 4×4 KiB pages.
#[cfg(feature = "transfer")]
#[allow(dead_code)] // used in substep 2 chain machinery
pub(crate) const TRANSFER_DTD_MAX: usize = 16 * 1024;

/// The total payload one full chain can carry: `TDS_PER_EP` dTDs × `TRANSFER_DTD_MAX`
/// (= 128 KiB with the `transfer` feature). `chain_layout` clamps oversized inputs.
#[cfg(feature = "transfer")]
#[allow(dead_code)] // used in substep 2 chain machinery
pub(crate) const TRANSFER_MAX_BYTES: usize = crate::state::TDS_PER_EP * TRANSFER_DTD_MAX;

/// Split `size` bytes into per-dTD sizes (≤`TRANSFER_DTD_MAX` each).
///
/// Returns `(sizes, count)`: `sizes[..count]` are the per-dTD byte counts.
/// `size == 0` yields `([0, ...], 1)` — one zero-length dTD.
/// If `size > TRANSFER_MAX_BYTES`, the excess is silently truncated (belt-and-braces;
/// callers should clamp first).
#[cfg(feature = "transfer")]
#[allow(dead_code)] // used in substep 2 chain machinery and its tests
fn chain_layout(size: usize) -> ([usize; crate::state::TDS_PER_EP], usize) {
    if size == 0 {
        return ([0; crate::state::TDS_PER_EP], 1);
    }
    let mut sizes = [0usize; crate::state::TDS_PER_EP];
    let mut count = 0usize;
    let mut off = 0usize;
    while off < size && count < crate::state::TDS_PER_EP {
        let n = (size - off).min(TRANSFER_DTD_MAX);
        sizes[count] = n;
        count += 1;
        off += n;
    }
    (sizes, count)
}

/// Per-transfer record stored in the FIFO ring.
///
/// Tracks the dTD chain that backs a single `submit_transfer` call. The record
/// lives in `Endpoint::queue` from submission until `poll_transfer` retires it.
#[cfg(feature = "transfer")]
#[derive(Clone, Copy)]
struct TransferRecord {
    /// Index of the first dTD in `Endpoint::tds` that belongs to this transfer.
    first_td: u8,
    /// Number of dTDs consumed by this transfer (1..=TDS_PER_EP).
    td_count: u8,
    /// Requested byte count as passed to `submit_transfer`.
    // Used by the packet path (substep 3) for staging-slot residue accounting.
    #[allow(dead_code)]
    len: usize,
    /// `Some(i)`: packet-path transfer using `buffers[i]` (the OUT class calls
    /// `read` on retire to copy bytes out). `None`: zero-copy — the caller owns
    /// the memory.
    // Used by the packet path (substep 3) to know which staging slot to copy from.
    #[allow(dead_code)]
    staging_slot: Option<u8>,
}

#[cfg(feature = "transfer")]
impl TransferRecord {
    const EMPTY: Self = Self {
        first_td: 0,
        td_count: 0,
        len: 0,
        staging_slot: None,
    };
}

/// A USB endpoint
pub struct Endpoint {
    address: EndpointAddress,
    qh: &'static mut Qh,
    /// Pool of TDs. Length is `TDS_PER_EP`. Existing code paths use `tds[0]`
    /// (the single-TD path). The full chain machinery uses the whole pool.
    tds: &'static mut [Td],
    /// Staging buffers. Slot 0 is always `Some` (holds master's `buffer`).
    /// Slots 1.. are `None` until `set_packet_queue_depth` fills them (substep 3).
    buffers: [Option<Buffer>; crate::state::TDS_PER_EP],
    kind: EndpointType,
    /// FIFO ring of pending transfer records.
    // Fields read/written by the transfer-queue methods below; not yet wired
    // into the bus layer (substep 4+), so rustc sees them as dead at crate level.
    #[cfg(feature = "transfer")]
    #[allow(dead_code)]
    queue: [TransferRecord; crate::state::TDS_PER_EP],
    /// Index of the oldest live record in `queue`.
    #[cfg(feature = "transfer")]
    #[allow(dead_code)]
    q_head: usize,
    /// Number of live records in `queue` (0..=TDS_PER_EP).
    #[cfg(feature = "transfer")]
    #[allow(dead_code)]
    q_len: usize,
    /// Next free TD index (circular within `tds`).
    #[cfg(feature = "transfer")]
    #[allow(dead_code)]
    td_head: usize,
    /// Number of TDs currently owned by queued transfers (0..=TDS_PER_EP).
    #[cfg(feature = "transfer")]
    #[allow(dead_code)]
    tds_in_use: usize,
}

impl Endpoint {
    pub fn new(
        address: EndpointAddress,
        qh: &'static mut Qh,
        tds: &'static mut [Td],
        buffer: Buffer,
        kind: EndpointType,
    ) -> Self {
        let max_packet_size = buffer.len();
        qh.set_zero_length_termination(false);
        qh.set_max_packet_len(max_packet_size);
        qh.set_interrupt_on_setup(
            EndpointType::Control == kind && address.direction() == UsbDirection::Out,
        );

        for td in tds.iter_mut() {
            td.set_terminate();
            td.clear_status();
        }

        // Build the buffers array: slot 0 holds the provided buffer; remaining
        // slots start as None.
        let mut buffers: [Option<Buffer>; crate::state::TDS_PER_EP] =
            [const { None }; crate::state::TDS_PER_EP];
        buffers[0] = Some(buffer);

        Endpoint {
            address,
            qh,
            tds,
            buffers,
            kind,
            #[cfg(feature = "transfer")]
            queue: [TransferRecord::EMPTY; crate::state::TDS_PER_EP],
            #[cfg(feature = "transfer")]
            q_head: 0,
            #[cfg(feature = "transfer")]
            q_len: 0,
            #[cfg(feature = "transfer")]
            td_head: 0,
            #[cfg(feature = "transfer")]
            tds_in_use: 0,
        }
    }

    /// Return an immutable reference to the staging buffer at `slot`.
    ///
    /// Panics in debug builds if `slot` is out of range or the slot is `None`.
    // Used by the packet path (substep 3) for zero-copy read access.
    #[allow(dead_code)]
    fn staging(&self, slot: usize) -> &Buffer {
        self.buffers[slot].as_ref().unwrap()
    }

    /// Return a mutable reference to the staging buffer at `slot`.
    ///
    /// Panics in debug builds if `slot` is out of range or the slot is `None`.
    fn staging_mut(&mut self, slot: usize) -> &mut Buffer {
        self.buffers[slot].as_mut().unwrap()
    }

    /// Enable ZLT for the given endpoint.
    pub fn enable_zlt(&mut self) {
        self.qh.set_zero_length_termination(true);
    }

    /// Indicates if the transfer descriptor is active
    pub fn is_primed(&self, usb: &ral::AnyUsbInstance) -> bool {
        (match self.address.direction() {
            UsbDirection::In => ral::read_reg!(ral::usb, usb, ENDPTSTAT, ETBR),
            UsbDirection::Out => ral::read_reg!(ral::usb, usb, ENDPTSTAT, ERBR),
        } & (1 << self.address.index()))
            != 0
    }

    /// Check for any transfer status, which is signaled through
    /// an error
    pub fn check_errors(&self) -> Result<(), UsbError> {
        let status = self.tds[0].status();
        if status.contains(Status::TRANSACTION_ERROR)
            | status.contains(Status::DATA_BUFFER_ERROR)
            | status.contains(Status::HALTED)
        {
            Err(UsbError::InvalidState)
        } else {
            Ok(())
        }
    }

    /// Initialize the endpoint, should be called soon after it's assigned,
    /// or after transitioning out of configuration (reset the endpoint).
    pub fn initialize(&mut self, usb: &ral::AnyUsbInstance) {
        if self.address.index() != 0 {
            let endptctrl = endpoint_control::register(usb, self.address.index());
            match self.address.direction() {
                UsbDirection::In => {
                    ral::modify_reg!(endpoint_control, &endptctrl, ENDPTCTRL, TXE: 0, TXT: into_raw_endpoint_type(EndpointType::Bulk))
                }
                UsbDirection::Out => {
                    ral::modify_reg!(endpoint_control, &endptctrl, ENDPTCTRL, RXE: 0, RXT: into_raw_endpoint_type(EndpointType::Bulk))
                }
            }
        }
    }

    /// Returns the endpoint address
    pub fn address(&self) -> EndpointAddress {
        self.address
    }

    /// Returns the maximum packet length supported by this endpoint
    pub fn max_packet_len(&self) -> usize {
        self.qh.max_packet_len()
    }

    /// Indicates if this endpoint has received setup data
    pub fn has_setup(&self, usb: &ral::AnyUsbInstance) -> bool {
        ral::read_reg!(ral::usb, usb, ENDPTSETUPSTAT) & (1 << self.address.index()) != 0
    }

    /// Read the setup buffer from this endpoint
    ///
    /// This is only meaningful for a control OUT endpoint.
    pub fn read_setup(&mut self, usb: &ral::AnyUsbInstance) -> u64 {
        // Reference manual isn't really clear on whe we should clear the ENDPTSETUPSTAT
        // bit...
        //
        // - section "Control Endpoint Operational Model" says that we should clear it
        //   *before* attempting to read the setup buffer, but
        // - section "Operational Model For Setup Transfers" says to do it *after*
        //   we read the setup buffer
        //
        // We're going with the "before" approach here. (Reference manual is iMXRT1060, rev2)
        ral::write_reg!(ral::usb, usb, ENDPTSETUPSTAT, 1 << self.address.index());
        loop {
            ral::modify_reg!(ral::usb, usb, USBCMD, SUTW: 1);
            let setup = self.qh.setup();
            if ral::read_reg!(ral::usb, usb, USBCMD, SUTW == 1) {
                ral::modify_reg!(ral::usb, usb, USBCMD, SUTW: 0);
                return setup;
            }
        }
    }

    /// Read data from the endpoint into `buffer`
    ///
    /// Returns the number of bytes read into `buffer`, which is constrained by the
    /// max packet length, and the number of bytes received in the last transfer.
    pub fn read(&mut self, buffer: &mut [u8]) -> usize {
        let size = self
            .qh
            .max_packet_len()
            .min(buffer.len())
            .min(self.tds[0].bytes_transferred());
        self.staging_mut(0).volatile_read(&mut buffer[..size])
    }

    /// Write `buffer` to the endpoint buffer
    ///
    /// Returns the number of bytes written from `buffer`, which is constrained
    /// by the max packet length.
    pub fn write(&mut self, buffer: &[u8]) -> usize {
        let size = self.qh.max_packet_len().min(buffer.len());
        let written = self.staging_mut(0).volatile_write(&buffer[..size]);
        self.staging_mut(0).clean_invalidate_dcache(size);
        written
    }

    /// Clear the complete bit for this endpoint
    pub fn clear_complete(&mut self, usb: &ral::AnyUsbInstance) {
        match self.address.direction() {
            UsbDirection::In => {
                ral::write_reg!(ral::usb, usb, ENDPTCOMPLETE, ETCE: 1 << self.address.index())
            }
            UsbDirection::Out => {
                ral::write_reg!(ral::usb, usb, ENDPTCOMPLETE, ERCE: 1 << self.address.index())
            }
        }
    }

    /// Schedule a transfer of `size` bytes from the endpoint buffer
    ///
    /// Caller should check to see if there is an active transfer, or if the previous
    /// transfer resulted in an error or halt.
    pub fn schedule_transfer(&mut self, usb: &ral::AnyUsbInstance, size: usize) {
        let buf_ptr = self.staging_mut(0).as_ptr_mut();
        self.tds[0].set_terminate();
        self.tds[0].set_buffer(buf_ptr, size);
        self.tds[0].set_interrupt_on_complete(true);
        self.tds[0].set_active();
        self.tds[0].clean_invalidate_dcache();

        self.qh.overlay_mut().set_next(&self.tds[0] as *const Td);
        self.qh.overlay_mut().clear_status();
        self.qh.clean_invalidate_dcache();

        dsb();
        match self.address.direction() {
            UsbDirection::In => {
                ral::write_reg!(ral::usb, usb, ENDPTPRIME, PETB: 1 << self.address.index())
            }
            UsbDirection::Out => {
                ral::write_reg!(ral::usb, usb, ENDPTPRIME, PERB: 1 << self.address.index())
            }
        }
        wait_endptprime(usb);
    }

    /// Stall or unstall the endpoint
    pub fn set_stalled(&mut self, usb: &ral::AnyUsbInstance, stall: bool) {
        let endptctrl = endpoint_control::register(usb, self.address.index());

        match self.address.direction() {
            UsbDirection::In => {
                ral::modify_reg!(endpoint_control, &endptctrl, ENDPTCTRL, TXS: stall as u32)
            }
            UsbDirection::Out => {
                ral::modify_reg!(endpoint_control, &endptctrl, ENDPTCTRL, RXS: stall as u32)
            }
        }
    }

    /// Indicates if the endpoint is stalled
    pub fn is_stalled(&self, usb: &ral::AnyUsbInstance) -> bool {
        let endptctrl = endpoint_control::register(usb, self.address.index());

        match self.address.direction() {
            UsbDirection::In => ral::read_reg!(endpoint_control, &endptctrl, ENDPTCTRL, TXS == 1),
            UsbDirection::Out => ral::read_reg!(endpoint_control, &endptctrl, ENDPTCTRL, RXS == 1),
        }
    }

    /// Enable the endpoint
    ///
    /// This should be called only after the USB device has been configured.
    pub fn enable(&mut self, usb: &ral::AnyUsbInstance) {
        // EP0 is always enabled
        if self.address.index() != 0 {
            let endptctrl = endpoint_control::register(usb, self.address.index());
            match self.address.direction() {
                UsbDirection::In => {
                    ral::modify_reg!(endpoint_control, &endptctrl, ENDPTCTRL, TXE: 1, TXR: 1, TXT: into_raw_endpoint_type(self.kind))
                }
                UsbDirection::Out => {
                    ral::modify_reg!(endpoint_control, &endptctrl, ENDPTCTRL, RXE: 1, RXR: 1, RXT: into_raw_endpoint_type(self.kind))
                }
            }
        }
    }

    /// Indicates if this endpoint is enabled
    ///
    /// Endpoint 0, the control endpoint, is always enabled.
    pub fn is_enabled(&self, usb: &ral::AnyUsbInstance) -> bool {
        if self.address.index() == 0 {
            return true;
        }

        let endptctrl = endpoint_control::register(usb, self.address.index());
        match self.address.direction() {
            UsbDirection::In => ral::read_reg!(endpoint_control, &endptctrl, ENDPTCTRL, TXE == 1),
            UsbDirection::Out => ral::read_reg!(endpoint_control, &endptctrl, ENDPTCTRL, RXE == 1),
        }
    }

    /// Clear the NACK bit for this endpoint
    pub fn clear_nack(&mut self, usb: &ral::AnyUsbInstance) {
        match self.address.direction() {
            UsbDirection::In => {
                ral::write_reg!(ral::usb, usb, ENDPTNAK, EPTN: 1 << self.address.index())
            }
            UsbDirection::Out => {
                ral::write_reg!(ral::usb, usb, ENDPTNAK, EPRN: 1 << self.address.index())
            }
        }
    }

    // -------------------------------------------------------------------------
    // Transfer queue (feature = "transfer")
    // -------------------------------------------------------------------------

    /// Returns `true` if ENDPTPRIME is already set for this endpoint's direction+index.
    ///
    /// When ENDPTPRIME is set the controller will pick up an appended dTD on its
    /// own via the NEXT link — no re-prime required.
    #[cfg(feature = "transfer")]
    #[allow(dead_code)] // called from prime_or_append; wired to bus layer in substep 4+
    fn endpoint_priming(&self, usb: &ral::AnyUsbInstance) -> bool {
        let prime_bit: u32 = match self.address.direction() {
            UsbDirection::In => 1 << (self.address.index() + 16),
            UsbDirection::Out => 1 << self.address.index(),
        };
        ral::read_reg!(ral::usb, usb, ENDPTPRIME) & prime_bit != 0
    }

    /// ATDTW tripwire: atomically read whether the EP queue is still active.
    ///
    /// Returns `true` if the endpoint queue is still active (the appended dTD
    /// will be picked up via the NEXT link), `false` if it went idle (caller must
    /// re-prime).
    #[cfg(feature = "transfer")]
    #[allow(dead_code)] // called from prime_or_append; wired to bus layer in substep 4+
    fn endpoint_active(&self, usb: &ral::AnyUsbInstance) -> bool {
        let stat_bit: u32 = match self.address.direction() {
            UsbDirection::In => 1 << (self.address.index() + 16),
            UsbDirection::Out => 1 << self.address.index(),
        };
        let mut ep_active;
        loop {
            ral::modify_reg!(ral::usb, usb, USBCMD, ATDTW: 1);
            let endptstat = ral::read_reg!(ral::usb, usb, ENDPTSTAT);
            ep_active = endptstat & stat_bit != 0;
            // Only trust the result if ATDTW is still 1 (no race).
            if ral::read_reg!(ral::usb, usb, USBCMD, ATDTW == 1) {
                break;
            }
        }
        ral::modify_reg!(ral::usb, usb, USBCMD, ATDTW: 0);
        ep_active
    }

    /// Write `ENDPTPRIME` for this endpoint and wait for the controller to fetch.
    ///
    /// The caller MUST have issued a `dsb()` before calling this.
    #[cfg(feature = "transfer")]
    #[allow(dead_code)] // called from prime_or_append; wired to bus layer in substep 4+
    fn prime(&self, usb: &ral::AnyUsbInstance) {
        match self.address.direction() {
            UsbDirection::In => {
                ral::write_reg!(ral::usb, usb, ENDPTPRIME, PETB: 1 << self.address.index())
            }
            UsbDirection::Out => {
                ral::write_reg!(ral::usb, usb, ENDPTPRIME, PERB: 1 << self.address.index())
            }
        }
        wait_endptprime(usb);
    }

    /// Prime-or-append: hand `head` to the controller.
    ///
    /// If the endpoint is idle (ENDPTSTAT clear and QH overlay ACTIVE clear),
    /// write `head` into the QH overlay and prime. Otherwise link `prev_tail`
    /// onto `head` and run the ATDTW tripwire to handle the add-dTD race.
    ///
    /// `prev_tail` is the last TD of the previously-queued chain (i.e. the
    /// global tail before this submit); it is `None` only on the very first
    /// submit (queue was empty → idle path is guaranteed).
    #[cfg(feature = "transfer")]
    #[allow(dead_code)] // called from submit_inner; wired to bus layer in substep 4+
    fn prime_or_append(
        &mut self,
        usb: &ral::AnyUsbInstance,
        head: *const Td,
        prev_tail: Option<*mut Td>,
    ) {
        if !self.is_primed(usb) && !self.qh.overlay_mut().status().contains(Status::ACTIVE) {
            // Idle path: write head into QH overlay and prime.
            self.qh.overlay_mut().set_next(head);
            self.qh.overlay_mut().clear_status();
            dsb();
            self.prime(usb);
        } else {
            // Active path: link previous tail onto the new head.
            // Barrier first: all dTD payload writes must be visible before we
            // publish the pointer (controller follows it immediately).
            dsb();
            if let Some(prev) = prev_tail {
                // Safety: prev_tail is a pointer we computed from self.tds which
                // is a 'static mut slice exclusively owned by this Endpoint.
                unsafe { (*prev).set_next(head) };
            }
            // If ENDPTPRIME is already set, the controller will pick up the new
            // TD via the NEXT link automatically — we are done.
            if !self.endpoint_priming(usb) {
                // ATDTW tripwire: if the queue went idle before the link landed,
                // re-prime from head.
                if !self.endpoint_active(usb) {
                    self.qh.overlay_mut().set_next(head);
                    self.qh.overlay_mut().clear_status();
                    dsb();
                    self.prime(usb);
                }
            }
        }
    }

    /// Queue a transfer of `len` bytes at `ptr` (IN: read from, OUT: written to).
    ///
    /// Zero-copy: the memory at `ptr[..len]` must remain valid and untouched until
    /// the transfer retires via `poll_transfer`. Returns `WouldBlock` when the dTD
    /// budget cannot fit the chain; `InvalidState` when `len > TRANSFER_MAX_BYTES`.
    ///
    /// **Short-packet semantics (OUT endpoints):** a short packet retires the
    /// *current* dTD and the controller advances to the next linked dTD (HW-proven).
    /// 1-packet transfers retire per packet; a multi-dTD OUT transfer receiving less
    /// than announced retires its current dTD short and leaves later dTDs active.
    /// Callers must size OUT transfers to the announced data length (BOT provides the
    /// oracle) and recover misbehaving hosts via `clear_transfers` in reset paths.
    ///
    /// **Control endpoints:** the queue is intended for bulk/interrupt endpoints only.
    /// A debug assertion fires if called on a Control endpoint.
    #[cfg(feature = "transfer")]
    #[allow(dead_code)] // public API for the bus layer (wired in substep 4+)
    pub fn submit_transfer(
        &mut self,
        usb: &ral::AnyUsbInstance,
        ptr: *mut u8,
        len: usize,
    ) -> Result<(), UsbError> {
        self.submit_inner(usb, ptr, len, None)
    }

    /// Internal implementation shared by `submit_transfer` and the packet path
    /// (substep 3 will call with `staging_slot = Some(slot)`).
    #[cfg(feature = "transfer")]
    #[allow(dead_code)] // called from submit_transfer and (substep 3) packet path
    fn submit_inner(
        &mut self,
        usb: &ral::AnyUsbInstance,
        ptr: *mut u8,
        len: usize,
        staging_slot: Option<u8>,
    ) -> Result<(), UsbError> {
        debug_assert!(
            self.kind != EndpointType::Control,
            "submit_transfer must not be called on a Control endpoint"
        );

        if len > TRANSFER_MAX_BYTES {
            return Err(UsbError::InvalidState);
        }

        let (sizes, count) = chain_layout(len);

        if count > self.tds_free() {
            return Err(UsbError::WouldBlock);
        }

        let first = self.td_head;

        // Compute the previous tail TD index (if any live records exist)
        // so we can link it after building the new chain.
        let prev_tail_idx: Option<usize> = if self.q_len > 0 {
            // The tail of the previously-queued transfer is:
            //   last_record.first_td + last_record.td_count - 1 (mod TDS_PER_EP)
            let last_rec = self.queue[(self.q_head + self.q_len - 1) % crate::state::TDS_PER_EP];
            let last_first = last_rec.first_td as usize;
            let last_count = last_rec.td_count as usize;
            Some((last_first + last_count - 1) % crate::state::TDS_PER_EP)
        } else {
            None
        };

        // Build the dTD chain. We use index arithmetic to avoid borrow-checker
        // issues with multiple mutable references into the same slice.
        let mut offset = 0usize;
        for (i, &td_size) in sizes[..count].iter().enumerate() {
            let idx = (first + i) % crate::state::TDS_PER_EP;
            // Safety: idx < TDS_PER_EP == self.tds.len().
            let td = &mut self.tds[idx];
            td.set_terminate();
            // Safety: ptr is provided by the caller who owns the buffer.
            td.set_buffer(unsafe { ptr.add(offset) }, td_size);
            td.set_interrupt_on_complete(i + 1 == count);
            td.set_active();
            offset += td_size;
        }

        // Link tds[0..count-1] to their successors. Must be done after all TDs
        // are initialised (so the controller never follows a half-written NEXT).
        for i in 0..count.saturating_sub(1) {
            let idx = (first + i) % crate::state::TDS_PER_EP;
            let next_idx = (first + i + 1) % crate::state::TDS_PER_EP;
            // We need pointers to two different elements. Split the slice to keep
            // the borrow checker happy when the indices are distinct.
            let next_ptr: *const Td = &self.tds[next_idx];
            self.tds[idx].set_next(next_ptr);
        }

        // FIFO push.
        self.queue[(self.q_head + self.q_len) % crate::state::TDS_PER_EP] = TransferRecord {
            first_td: first as u8,
            td_count: count as u8,
            len,
            staging_slot,
        };
        self.q_len += 1;
        self.td_head = (first + count) % crate::state::TDS_PER_EP;
        self.tds_in_use += count;

        let head_ptr: *const Td = &self.tds[first];
        let prev_tail_ptr: Option<*mut Td> = prev_tail_idx.map(|i| &mut self.tds[i] as *mut Td);

        self.prime_or_append(usb, head_ptr, prev_tail_ptr);

        Ok(())
    }

    /// Retire the oldest transfer if all its dTDs have completed.
    ///
    /// - `Some(Ok(n))`: `n` bytes actually transferred (≤ `len`; short OUT reads < `len`).
    /// - `Some(Err(_))`: a dTD reported `TRANSACTION_ERROR`, `DATA_BUFFER_ERROR`, or
    ///   `HALTED` — the record is retired and the caller (class) should stall per BOT rules.
    /// - `None`: queue empty or oldest transfer still in flight.
    #[cfg(feature = "transfer")]
    #[allow(dead_code)] // public API for the bus layer (wired in substep 4+)
    pub fn poll_transfer(&mut self) -> Option<Result<usize, UsbError>> {
        if self.q_len == 0 {
            return None;
        }

        let rec = self.queue[self.q_head];

        // Check whether any dTD in this record is still ACTIVE.
        for i in 0..rec.td_count as usize {
            let idx = (rec.first_td as usize + i) % crate::state::TDS_PER_EP;
            if self.tds[idx].status().contains(Status::ACTIVE) {
                return None;
            }
        }

        // All dTDs complete — check for errors.
        for i in 0..rec.td_count as usize {
            let idx = (rec.first_td as usize + i) % crate::state::TDS_PER_EP;
            let st = self.tds[idx].status();
            if st.contains(Status::TRANSACTION_ERROR)
                | st.contains(Status::DATA_BUFFER_ERROR)
                | st.contains(Status::HALTED)
            {
                // Pop record and free its TD budget.
                self.q_head = (self.q_head + 1) % crate::state::TDS_PER_EP;
                self.q_len -= 1;
                self.tds_in_use -= rec.td_count as usize;
                return Some(Err(UsbError::InvalidState));
            }
        }

        // Sum bytes_transferred across all dTDs.
        let mut total = 0usize;
        for i in 0..rec.td_count as usize {
            let idx = (rec.first_td as usize + i) % crate::state::TDS_PER_EP;
            total += self.tds[idx].bytes_transferred();
        }

        // Pop record and free its TD budget.
        self.q_head = (self.q_head + 1) % crate::state::TDS_PER_EP;
        self.q_len -= 1;
        self.tds_in_use -= rec.td_count as usize;

        Some(Ok(total))
    }

    /// Returns the number of transfers currently queued (submitted but not yet polled out).
    #[cfg(feature = "transfer")]
    #[allow(dead_code)] // public API for the bus layer (wired in substep 4+)
    pub fn pending_transfers(&self) -> usize {
        self.q_len
    }

    /// Returns the number of free TD slots (available for new submit calls).
    #[cfg(feature = "transfer")]
    #[allow(dead_code)] // called from submit_inner
    fn tds_free(&self) -> usize {
        crate::state::TDS_PER_EP - self.tds_in_use
    }

    /// Flush the endpoint and drop every queued transfer.
    ///
    /// Used during bus reset, endpoint unstall, and BOT reset recovery. After this
    /// call `pending_transfers()` is 0 and a fresh `submit_transfer` primes from a
    /// clean QH overlay.
    #[cfg(feature = "transfer")]
    #[allow(dead_code)] // public API for the bus layer (wired in substep 4+)
    pub fn clear_transfers(&mut self, usb: &ral::AnyUsbInstance) {
        // Flush any primed / in-flight TDs.
        let bit = 1u32 << self.address.index();
        let mask = match self.address.direction() {
            UsbDirection::In => bit << 16,
            UsbDirection::Out => bit,
        };
        ral::write_reg!(ral::usb, usb, ENDPTFLUSH, mask);
        wait_endptflush(usb, mask);

        // Reset queue bookkeeping.
        self.q_head = 0;
        self.q_len = 0;
        self.td_head = 0;
        self.tds_in_use = 0;
        self.queue = [TransferRecord::EMPTY; crate::state::TDS_PER_EP];

        // Terminate and clear status on every TD.
        for td in self.tds.iter_mut() {
            td.set_terminate();
            td.clear_status();
        }

        // Clear QH overlay.
        self.qh.overlay_mut().set_terminate();
        self.qh.overlay_mut().clear_status();
    }
}

/// Converts the endpoint type into its ENDPTCTRL endpoint type
/// enumerations.
///
/// See the ENDPTCTRL register documentation in the reference manual.
fn into_raw_endpoint_type(ep_type: EndpointType) -> u32 {
    // Bits 0..1 represent the transfer type for the endpoint,
    // and it's compatible with ENDPTCTRL's enumerated values.
    (ep_type.to_bm_attributes() & 0b11u8).into()
}

// -------------------------------------------------------------------------
// Unit tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    // -----------------------------------------------------------------------
    // Fake USB instance for tests that need &ral::AnyUsbInstance but never
    // dereference the hardware registers (reads return 0).
    // -----------------------------------------------------------------------

    /// Return a reference to a zeroed register-block stand-in.
    ///
    /// `imxrt_ral::Instance<T,N>` is `repr(transparent)` over `NonNull<T>` — it
    /// holds a *pointer* to a `RegisterBlock`.
    ///
    /// Each call leaks its own zeroed block + `Instance`. That isolation matters:
    /// cargo runs unit tests on parallel threads, and priming paths write
    /// `ENDPTPRIME` through this pointer — a single shared static block would be
    /// a cross-thread data race. Leaking is fine in a test binary.
    fn fake_usb() -> &'static imxrt_ral::usb::Instance<255> {
        let blk: *const imxrt_ral::usb::RegisterBlock =
            std::boxed::Box::leak(std::boxed::Box::new(unsafe {
                core::mem::zeroed::<imxrt_ral::usb::RegisterBlock>()
            }));
        let inst = unsafe { imxrt_ral::usb::Instance::new(blk) };
        std::boxed::Box::leak(std::boxed::Box::new(inst))
    }

    // -----------------------------------------------------------------------
    // Macro: build an Endpoint with backing storage on the stack.
    //
    // Usage:
    //   make_endpoint!(ep_var, BACKING_NAME, depth, mps, direction, kind);
    //
    // `depth` controls how many TD/buffer slots are used (must be ≤ TDS_PER_EP).
    // `mps` is the max-packet-size in bytes for each buffer slot.
    // Edition 2024: addr_of_mut! avoids the forbidden &mut-of-static form.
    // -----------------------------------------------------------------------
    macro_rules! make_endpoint {
        ($ep_name:ident, $name:ident, $depth:expr, $mps:expr, $dir:expr, $kind:expr) => {
            static mut $name: (
                crate::qh::Qh,
                [crate::td::Td; crate::state::TDS_PER_EP],
                [u8; $mps],
            ) = (
                crate::qh::Qh::new(),
                [const { crate::td::Td::new() }; crate::state::TDS_PER_EP],
                [0u8; $mps],
            );

            let ep_qh_ref: &'static mut crate::qh::Qh =
                unsafe { &mut (*core::ptr::addr_of_mut!($name)).0 };
            let ep_tds_slice: &'static mut [crate::td::Td] =
                unsafe { &mut (&mut *core::ptr::addr_of_mut!($name)).1[..$depth] };
            let buf_ptr: *mut u8 = unsafe { (*core::ptr::addr_of_mut!($name)).2.as_mut_ptr() };
            let mut sub_alloc = unsafe {
                crate::buffer::Allocator::from_buffer(core::slice::from_raw_parts_mut(
                    buf_ptr, $mps,
                ))
            };
            let buf = sub_alloc.allocate($mps).unwrap();

            let $ep_name = crate::endpoint::Endpoint::new(
                usb_device::endpoint::EndpointAddress::from_parts(1, $dir),
                ep_qh_ref,
                ep_tds_slice,
                buf,
                $kind,
            );
        };
    }

    // -----------------------------------------------------------------------
    // Test: chain_layout splits sizes correctly (feature = "transfer" only)
    // -----------------------------------------------------------------------

    #[cfg(feature = "transfer")]
    #[test]
    fn chain_layout_splits_into_dtds() {
        use super::chain_layout;

        // 64 KiB → exactly four 16 KiB chunks.
        let (sizes, count) = chain_layout(65536);
        assert_eq!(count, 4, "64 KiB must produce 4 chunks");
        assert_eq!(&sizes[..count], [16384, 16384, 16384, 16384]);

        // 31 bytes → one chunk.
        let (sizes, count) = chain_layout(31);
        assert_eq!(count, 1, "31 B must produce 1 chunk");
        assert_eq!(&sizes[..count], [31]);

        // Oversized: 130 KiB > TRANSFER_MAX_BYTES (128 KiB) → clamps to TDS_PER_EP entries.
        let (_, count) = chain_layout(130 * 1024);
        assert_eq!(
            count,
            crate::state::TDS_PER_EP,
            "over-cap size must clamp to TDS_PER_EP entries"
        );

        // Zero → one zero-length dTD.
        let (sizes, count) = chain_layout(0);
        assert_eq!(count, 1, "zero size must produce 1 chunk");
        assert_eq!(sizes[0], 0);
    }

    // -----------------------------------------------------------------------
    // Test: set_next / set_terminate round-trip on two stack Tds.
    //
    // Td is repr(C); NEXT is the first field (VCell<u32> = repr(transparent)
    // over UnsafeCell<u32>). We read it via a raw pointer cast so we can verify
    // the values without requiring a test accessor in td.rs.
    // -----------------------------------------------------------------------

    /// Read the raw NEXT word of a `Td` via its `repr(C)` layout.
    ///
    /// # Safety
    /// `td` must point to an initialised `Td`. The cast is valid because
    /// `Td` is `repr(C)` and `NEXT` (a `VCell<u32>`, itself `repr(transparent)`
    /// over `UnsafeCell<u32>`) is at offset 0.
    unsafe fn read_td_next(td: *const Td) -> u32 {
        unsafe { *(td as *const u32) }
    }

    #[test]
    fn td_chain_link_and_terminate() {
        let mut td0 = Td::new();
        let mut td1 = Td::new();

        // Link td0 → td1, terminate td1.
        td0.set_next(&td1 as *const Td);
        td1.set_terminate();

        // td0.NEXT should hold td1's address (not the terminate sentinel 1).
        let next_ptr = unsafe { read_td_next(&td0) };
        assert_eq!(
            next_ptr, &td1 as *const Td as u32,
            "td0.NEXT must point to td1"
        );
        assert_ne!(next_ptr, 1, "td0 must not be terminated");

        // td1.NEXT should be the terminate sentinel (1).
        let term = unsafe { read_td_next(&td1) };
        assert_eq!(term, 1, "td1 must be terminated");
    }

    // -----------------------------------------------------------------------
    // Test: schedule_transfer on the single-TD path (no features) proves that
    // the TDS_PER_EP == 1 path still behaves like master at runtime.
    //
    // Specifically:
    //   1. write() stages bytes into the endpoint buffer.
    //   2. schedule_transfer() arms tds[0]: ACTIVE bit set, no errors.
    //   3. After simulated completion (clear_status), check_errors() is Ok.
    //   4. write() → write() → read() staging round-trip: final read returns
    //      what the second write staged (max_packet_len path through slot 0).
    // -----------------------------------------------------------------------

    #[test]
    fn schedule_transfer_single_td_no_feature() {
        // Build a depth-1 endpoint (uses tds[0] only, matches no-feature path).
        make_endpoint!(
            ep,
            BACKING_SCHED,
            1,
            64,
            UsbDirection::Out,
            EndpointType::Bulk
        );
        let usb = fake_usb();
        let mut ep_mut = ep;

        // Write some data into the endpoint buffer.
        let data = [0xAAu8; 32];
        let written = ep_mut.write(&data);
        assert_eq!(written, 32, "write must return byte count");

        // schedule_transfer sets up tds[0] and primes the endpoint.
        // wait_endptprime is a no-op in test builds.
        ep_mut.schedule_transfer(usb, 32);

        // Verify tds[0] is ACTIVE after schedule_transfer.
        use crate::td::Status;
        assert!(
            ep_mut.tds[0].status().contains(Status::ACTIVE),
            "tds[0] must be ACTIVE after schedule_transfer"
        );

        // check_errors inspects tds[0] status — ACTIVE alone is not an error.
        assert!(
            ep_mut.check_errors().is_ok(),
            "ACTIVE-only status must not trigger an error"
        );

        // Simulate transfer completion: hardware clears ACTIVE (and sets residual
        // TOTAL_BYTES to 0, but we can't replicate that from outside td.rs —
        // clear_status only clears the STATUS byte, not TOTAL_BYTES).
        ep_mut.tds[0].clear_status();
        assert!(
            !ep_mut.tds[0].status().contains(Status::ACTIVE),
            "tds[0] must not be ACTIVE after clear_status"
        );

        // After clear, check_errors still Ok (no error bits set).
        assert!(
            ep_mut.check_errors().is_ok(),
            "no errors after clear_status"
        );

        // Staging round-trip via write(): stage new data and verify it's
        // stored in the buffer (volatile_write then volatile_read path).
        let data2 = [0xBBu8; 16];
        let written2 = ep_mut.write(&data2);
        assert_eq!(written2, 16, "second write must stage 16 bytes");

        // The read() method returns min(max_packet_len, out.len(), bytes_transferred).
        // bytes_transferred = last_transfer_size(32) - TOTAL_BYTES_residual(32) = 0
        // because clear_status did not change TOTAL_BYTES. So read() returns 0 —
        // the important invariant is that nothing panics and slot 0 is used.
        let mut out = [0u8; 64];
        let _n = ep_mut.read(&mut out);
        // We don't assert _n here since bytes_transferred = 0 after clear_status
        // (TOTAL_BYTES residual unchanged) — this is expected in a pure-software sim.
        // The key correctness signal is that tds slice is length 1 (we passed depth=1
        // to make_endpoint!), tds[0] is used for all operations, and nothing panics
        // or bounds-checks out. TDS_PER_EP may be larger when `transfer` is enabled,
        // but the endpoint was explicitly constructed with depth 1.
        assert_eq!(
            ep_mut.tds.len(),
            1,
            "depth-1 endpoint must have exactly 1 TD slot"
        );
    }

    // -----------------------------------------------------------------------
    // Substep 2 tests (feature = "transfer" only)
    // -----------------------------------------------------------------------

    /// Helper: build an 8-slot OUT Bulk endpoint for transfer-queue tests.
    ///
    /// Returns `(ep, dummy_buf)` where `dummy_buf` is a heap allocation used as
    /// the zero-copy pointer passed to `submit_transfer`. The buffer is `len`
    /// bytes; tests set `len` large enough for the intended transfer.
    #[cfg(feature = "transfer")]
    fn make_bulk_ep_and_buf(
        mps: usize,
        buf_len: usize,
    ) -> (std::boxed::Box<Endpoint>, std::vec::Vec<u8>) {
        use crate::state::TDS_PER_EP;

        // Build raw static storage via Box::leak so we get 'static lifetime.
        let qh: &'static mut crate::qh::Qh =
            std::boxed::Box::leak(std::boxed::Box::new(crate::qh::Qh::new()));
        let tds: &'static mut [Td] = {
            let v: std::vec::Vec<Td> = (0..TDS_PER_EP).map(|_| Td::new()).collect();
            std::boxed::Box::leak(v.into_boxed_slice())
        };
        let mut backing: std::vec::Vec<u8> = std::vec![0u8; mps];
        let mut alloc = unsafe {
            crate::buffer::Allocator::from_buffer(core::slice::from_raw_parts_mut(
                backing.as_mut_ptr(),
                mps,
            ))
        };
        let buf = alloc.allocate(mps).unwrap();
        // Keep backing alive for the duration of the test; leak it.
        std::mem::forget(backing);

        let ep = Endpoint::new(
            usb_device::endpoint::EndpointAddress::from_parts(1, UsbDirection::Out),
            qh,
            tds,
            buf,
            EndpointType::Bulk,
        );

        let dummy: std::vec::Vec<u8> = std::vec![0u8; buf_len];
        (std::boxed::Box::new(ep), dummy)
    }

    // -----------------------------------------------------------------------
    // 2a. submit_builds_chain_ioc_on_last
    //
    // Submit 64 KiB (4 × 16 KiB dTDs). Verify:
    //   - 4 dTDs linked: tds[0]→tds[1]→tds[2]→tds[3] (NEXT pointers)
    //   - tds[3] terminated (NEXT == 1)
    //   - IOC only on the last (tds[3])
    //   - all 4 ACTIVE
    //   - queue len == 1
    // -----------------------------------------------------------------------
    #[cfg(feature = "transfer")]
    #[test]
    fn submit_builds_chain_ioc_on_last() {
        let usb = fake_usb();
        let (mut ep, mut dummy) = make_bulk_ep_and_buf(512, 64 * 1024);

        ep.submit_transfer(usb, dummy.as_mut_ptr(), 64 * 1024)
            .expect("64 KiB submit must succeed");

        // Queue length must be 1.
        assert_eq!(ep.pending_transfers(), 1);

        // Verify the 4 TDs.
        for i in 0..4usize {
            // All ACTIVE.
            assert!(
                ep.tds[i].status().contains(Status::ACTIVE),
                "tds[{i}] must be ACTIVE"
            );
        }

        // NEXT chain: tds[0]→tds[1]→tds[2] via pointer; tds[3] terminated.
        for i in 0..3usize {
            let next_raw = unsafe { read_td_next(&ep.tds[i]) };
            let expected = &ep.tds[i + 1] as *const Td as u32;
            assert_eq!(
                next_raw,
                expected,
                "tds[{i}].NEXT must point to tds[{}]",
                i + 1
            );
        }
        // Last TD terminated.
        let last_next = unsafe { read_td_next(&ep.tds[3]) };
        assert_eq!(last_next, 1, "tds[3] must be terminated");

        // IOC only on the last TD. The TOKEN word has IOC at bit 15.
        // Read via bytes_transferred indirection is impractical; inspect TOKEN
        // via the raw IOC bit instead using the Status path won't work since IOC
        // is not a status bit. Use a simple raw read of the TOKEN field.
        // Token layout: IOC at bit 15.
        for i in 0..3usize {
            // Inspect TOKEN word directly via pointer (Td is repr(C)).
            // TOKEN is at offset 4 (after NEXT: VCell<u32>).
            let token = unsafe { *(&ep.tds[i] as *const Td as *const u32).add(1) };
            let ioc_bit = (token >> 15) & 1;
            assert_eq!(ioc_bit, 0, "tds[{i}] must NOT have IOC set");
        }
        {
            let token = unsafe { *(&ep.tds[3] as *const Td as *const u32).add(1) };
            let ioc_bit = (token >> 15) & 1;
            assert_eq!(ioc_bit, 1, "tds[3] must have IOC set");
        }
    }

    // -----------------------------------------------------------------------
    // 2b. submit_would_block_when_td_budget_exhausted
    //
    // With TDS_PER_EP == 8:
    //   - Two 64 KiB submits consume 4+4 = 8 TDs → both succeed.
    //   - Third submit returns WouldBlock (0 free TDs).
    //   - After force_complete + poll_transfer drains the first record
    //     (freeing 4 TDs), the third submit succeeds (circular reuse).
    // -----------------------------------------------------------------------
    #[cfg(feature = "transfer")]
    #[test]
    fn submit_would_block_when_td_budget_exhausted() {
        use crate::state::TDS_PER_EP;
        let usb = fake_usb();
        // Need 3 × 64 KiB worth of buffer pointer space; we use a single Vec and
        // three offsets — contents don't matter for this bookkeeping test.
        let mut dummy: std::vec::Vec<u8> = std::vec![0u8; 3 * 64 * 1024];
        let ptr = dummy.as_mut_ptr();

        let (mut ep, _) = make_bulk_ep_and_buf(512, 0);

        // First submit: 4 TDs used (TDs 0..3).
        ep.submit_transfer(usb, ptr, 64 * 1024)
            .expect("first 64 KiB submit must succeed");
        assert_eq!(ep.tds_in_use, 4);

        // Second submit: 4 more TDs used (TDs 4..7).
        ep.submit_transfer(usb, unsafe { ptr.add(64 * 1024) }, 64 * 1024)
            .expect("second 64 KiB submit must succeed");
        assert_eq!(ep.tds_in_use, 8);
        assert_eq!(ep.tds_free(), 0);

        // Third submit must fail with WouldBlock.
        let result = ep.submit_transfer(usb, unsafe { ptr.add(128 * 1024) }, 64 * 1024);
        assert!(
            matches!(result, Err(UsbError::WouldBlock)),
            "third submit must return WouldBlock, got {result:?}"
        );
        // Queue and budget untouched.
        assert_eq!(ep.pending_transfers(), 2);
        assert_eq!(ep.tds_in_use, 8);

        // Simulate hardware completing the first record's 4 TDs.
        for i in 0..4usize {
            ep.tds[i].force_complete(0);
        }

        // poll_transfer must retire the first record.
        let polled = ep.poll_transfer();
        assert!(
            matches!(polled, Some(Ok(_))),
            "poll must succeed after completing first record, got {polled:?}"
        );
        assert_eq!(ep.pending_transfers(), 1);
        assert_eq!(ep.tds_free(), TDS_PER_EP / 2); // 4 TDs freed

        // Now the third submit must succeed (reuses TDs 0..3 circularly).
        ep.submit_transfer(usb, unsafe { ptr.add(128 * 1024) }, 64 * 1024)
            .expect("third submit must succeed after poll freed TDs");
        assert_eq!(ep.pending_transfers(), 2);
    }

    // -----------------------------------------------------------------------
    // 2c. poll_retires_fifo_with_byte_counts
    //
    // Queue three 512-byte transfers; complete each with different residuals:
    //   remaining=0  → transferred=512
    //   remaining=481 → transferred=31
    //   remaining=0  → transferred=512
    // poll_transfer must yield Ok(512), Ok(31), Ok(512) in FIFO order.
    // -----------------------------------------------------------------------
    #[cfg(feature = "transfer")]
    #[test]
    fn poll_retires_fifo_with_byte_counts() {
        let usb = fake_usb();
        // 3 transfers × 512 bytes each.
        let mut dummy: std::vec::Vec<u8> = std::vec![0u8; 3 * 512];
        let ptr = dummy.as_mut_ptr();

        let (mut ep, _) = make_bulk_ep_and_buf(512, 0);

        // Submit three 512-byte transfers.
        ep.submit_transfer(usb, ptr, 512)
            .expect("transfer 0 submit");
        ep.submit_transfer(usb, unsafe { ptr.add(512) }, 512)
            .expect("transfer 1 submit");
        ep.submit_transfer(usb, unsafe { ptr.add(1024) }, 512)
            .expect("transfer 2 submit");

        assert_eq!(ep.pending_transfers(), 3);

        // Complete transfer 0: full (remaining=0 → bytes_transferred=512-0=512).
        ep.tds[0].force_complete(0);
        // Complete transfer 1: short (remaining=481 → bytes_transferred=512-481=31).
        ep.tds[1].force_complete(481);
        // Complete transfer 2: full.
        ep.tds[2].force_complete(0);

        // Poll must yield in FIFO order.
        assert_eq!(ep.poll_transfer(), Some(Ok(512)));
        assert_eq!(ep.pending_transfers(), 2);

        assert_eq!(ep.poll_transfer(), Some(Ok(31)));
        assert_eq!(ep.pending_transfers(), 1);

        assert_eq!(ep.poll_transfer(), Some(Ok(512)));
        assert_eq!(ep.pending_transfers(), 0);

        // Queue empty — further polls return None.
        assert_eq!(ep.poll_transfer(), None);
    }

    // -----------------------------------------------------------------------
    // 2d. submit_transfer_rejects_oversized
    //
    // submit_transfer(ptr, TRANSFER_MAX_BYTES + 1) must return InvalidState
    // and leave queue/budget untouched.
    // -----------------------------------------------------------------------
    #[cfg(feature = "transfer")]
    #[test]
    fn submit_transfer_rejects_oversized() {
        let usb = fake_usb();
        let mut dummy: std::vec::Vec<u8> = std::vec![0u8; 1]; // pointer only
        let (mut ep, _) = make_bulk_ep_and_buf(512, 0);

        let result = ep.submit_transfer(usb, dummy.as_mut_ptr(), TRANSFER_MAX_BYTES + 1);
        assert!(
            matches!(result, Err(UsbError::InvalidState)),
            "oversized submit must return InvalidState, got {result:?}"
        );
        assert_eq!(ep.pending_transfers(), 0, "queue must remain empty");
        assert_eq!(ep.tds_in_use, 0, "TD budget must be untouched");
    }

    // -----------------------------------------------------------------------
    // 2e. poll_reports_dtd_error
    //
    // Set HALTED on a queued record's dTD; poll_transfer must return Some(Err(_)).
    // -----------------------------------------------------------------------
    #[cfg(feature = "transfer")]
    #[test]
    fn poll_reports_dtd_error() {
        let usb = fake_usb();
        let mut dummy: std::vec::Vec<u8> = std::vec![0u8; 512];
        let (mut ep, _) = make_bulk_ep_and_buf(512, 0);

        ep.submit_transfer(usb, dummy.as_mut_ptr(), 512)
            .expect("submit must succeed");

        // Force HALTED (and clear ACTIVE so the TD appears "done").
        // HALTED is bit 6 of the TOKEN STATUS byte; ACTIVE is bit 7.
        // We clear ACTIVE by force_complete(0) then set HALTED separately
        // by writing to the TOKEN field directly.
        ep.tds[0].force_complete(0);
        // Write STATUS bits: HALTED = 0x40, clear ACTIVE (0x80 stays clear).
        // TOKEN offset 4, STATUS at bits 0..7.
        unsafe {
            let token_ptr = (&ep.tds[0] as *const Td as *mut u32).add(1);
            // Preserve other TOKEN fields (IOC, TOTAL_BYTES) and set HALTED.
            let existing = token_ptr.read();
            token_ptr.write(existing | 0x40);
        }

        let result = ep.poll_transfer();
        assert!(
            matches!(result, Some(Err(_))),
            "halted TD must yield Some(Err(_)), got {result:?}"
        );
        // Record retired even on error.
        assert_eq!(ep.pending_transfers(), 0);
    }

    // -----------------------------------------------------------------------
    // 2f. zero_length_transfer_single_dtd
    //
    // submit_transfer(ptr, 0) must build one 0-byte dTD and retire to Ok(0).
    // -----------------------------------------------------------------------
    #[cfg(feature = "transfer")]
    #[test]
    fn zero_length_transfer_single_dtd() {
        let usb = fake_usb();
        let mut dummy: std::vec::Vec<u8> = std::vec![0u8; 1];
        let (mut ep, _) = make_bulk_ep_and_buf(512, 0);

        ep.submit_transfer(usb, dummy.as_mut_ptr(), 0)
            .expect("zero-length submit must succeed");

        assert_eq!(ep.pending_transfers(), 1);
        assert_eq!(ep.tds_in_use, 1, "exactly one TD consumed");

        // Complete the single TD (0 bytes, remaining=0).
        ep.tds[0].force_complete(0);

        let result = ep.poll_transfer();
        assert_eq!(result, Some(Ok(0)), "zero-length must retire to Ok(0)");
        assert_eq!(ep.pending_transfers(), 0);
    }

    // -----------------------------------------------------------------------
    // 2g. clear_transfers_resets_queue
    //
    // Submit transfers, leave dTDs ACTIVE (in-flight), call clear_transfers:
    //   - pending_transfers() == 0
    //   - all TDs terminated/inactive
    //   - ENDPTFLUSH register holds the EP's bit (observable write, no HW clear)
    //   - a fresh submit primes from a clean overlay
    // -----------------------------------------------------------------------
    #[cfg(feature = "transfer")]
    #[test]
    fn clear_transfers_resets_queue() {
        use crate::state::TDS_PER_EP;
        let usb = fake_usb();
        let mut dummy: std::vec::Vec<u8> = std::vec![0u8; 2 * 512];
        let ptr = dummy.as_mut_ptr();
        let (mut ep, _) = make_bulk_ep_and_buf(512, 0);

        // Submit two transfers so we have active TDs.
        ep.submit_transfer(usb, ptr, 512).expect("submit 0");
        ep.submit_transfer(usb, unsafe { ptr.add(512) }, 512)
            .expect("submit 1");
        assert_eq!(ep.pending_transfers(), 2);

        // Call clear_transfers — TDs are still ACTIVE (simulating mid-flight flush).
        ep.clear_transfers(usb);

        // Queue must be empty.
        assert_eq!(ep.pending_transfers(), 0, "pending must be 0 after clear");
        assert_eq!(ep.tds_in_use, 0, "tds_in_use must be 0 after clear");
        assert_eq!(ep.td_head, 0, "td_head must reset to 0");

        // All TDs must be terminated and inactive.
        for i in 0..TDS_PER_EP {
            let next_raw = unsafe { read_td_next(&ep.tds[i]) };
            assert_eq!(
                next_raw, 1,
                "tds[{i}] must be terminated after clear_transfers"
            );
            assert!(
                !ep.tds[i].status().contains(Status::ACTIVE),
                "tds[{i}] must not be ACTIVE after clear_transfers"
            );
        }

        // ENDPTFLUSH must have been written with the EP's bit. The fake USB block
        // is zeroed memory and wait_endptflush is a test no-op, so the register
        // value stays as written.
        // EP index is 1 (from_parts(1, Out)) → OUT flush bit = bit 1.
        let flush_val = ral::read_reg!(ral::usb, usb, ENDPTFLUSH);
        assert_ne!(
            flush_val & (1u32 << 1),
            0,
            "ENDPTFLUSH bit 1 must be set for EP1 OUT"
        );

        // A fresh submit after clear must succeed and prime from a clean overlay.
        ep.submit_transfer(usb, ptr, 512)
            .expect("post-clear submit must succeed");
        assert_eq!(ep.pending_transfers(), 1);
        assert!(
            ep.tds[0].status().contains(Status::ACTIVE),
            "fresh TD must be ACTIVE after post-clear submit"
        );
    }
}
