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
    state::RING_DEPTH,
    td::{Status, Td},
};
use usb_device::{
    UsbDirection, UsbError,
    endpoint::{EndpointAddress, EndpointType},
};

/// A USB endpoint with a multi-TD ring per direction.
pub struct Endpoint {
    address: EndpointAddress,
    qh: &'static mut Qh,
    /// Slice of length RING_DEPTH. Only `depth()` slots are used.
    tds: &'static mut [Td],
    /// `len()` == `depth()`: 1 for control/interrupt, RING_DEPTH for bulk.
    buffers: heapless::Vec<Buffer, RING_DEPTH>,
    kind: EndpointType,
    /// Next slot to prime (0..depth()).
    head: usize,
    /// Oldest in-flight slot (0..depth()).
    tail: usize,
    /// Number of TDs primed but not yet retired (0..=depth()).
    in_flight: u32,
}

impl Endpoint {
    pub fn new(
        address: EndpointAddress,
        qh: &'static mut Qh,
        tds: &'static mut [Td],
        buffers: heapless::Vec<Buffer, RING_DEPTH>,
        kind: EndpointType,
    ) -> Self {
        let max_packet_size = buffers[0].len();
        qh.set_zero_length_termination(false);
        qh.set_max_packet_len(max_packet_size);
        qh.set_interrupt_on_setup(
            EndpointType::Control == kind && address.direction() == UsbDirection::Out,
        );

        for td in tds.iter_mut().take(buffers.len()) {
            td.set_terminate();
            td.clear_status();
        }

        Endpoint {
            address,
            qh,
            tds,
            buffers,
            kind,
            head: 0,
            tail: 0,
            in_flight: 0,
        }
    }

    /// Number of active ring slots (== buffers.len()).
    fn depth(&self) -> usize {
        self.buffers.len()
    }

    /// Returns `true` when all ring slots are primed (no free slot to add).
    pub fn all_busy(&self) -> bool {
        self.in_flight as usize == self.depth()
    }

    /// Returns `true` when the ring holds no in-flight dTDs.
    ///
    /// For an IN endpoint this means every primed transfer has been delivered to
    /// the host and drained by [`complete`](Endpoint::complete). BOT uses this to
    /// serialize the CSW: the IN ring must be empty at a command boundary, or the
    /// host can read a stale/duplicate response (cross-transaction pipelining is
    /// only safe *within* a single command's data phase).
    pub fn ring_drained(&self) -> bool {
        self.in_flight == 0
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
    /// an error.
    ///
    /// Inspects slot 0 (legacy/control/Lever-A path).
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
        // A bus reset flushes the controller's primed TDs (ENDPTFLUSH). Reset the
        // ring bookkeeping to match, or stale `in_flight` from a prior session
        // desyncs the ring and eventually wedges it (`all_busy()` never clears →
        // perpetual WouldBlock).
        self.reset_ring();
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
        // Reference manual isn't really clear on when we should clear the ENDPTSETUPSTAT
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

    /// Read data from the endpoint into `buffer` (legacy/EP0 path).
    ///
    /// Returns the number of bytes read into `buffer`, constrained by the
    /// max packet length and the bytes received in the last transfer (slot 0).
    pub fn read(&mut self, buffer: &mut [u8]) -> usize {
        let size = self
            .qh
            .max_packet_len()
            .min(buffer.len())
            .min(self.tds[0].bytes_transferred());
        self.buffers[0].volatile_read(&mut buffer[..size])
    }

    /// Write `buffer` to the endpoint buffer (legacy/EP0 path, slot 0).
    ///
    /// Returns the number of bytes written from `buffer`, constrained
    /// by the max packet length.
    pub fn write(&mut self, buffer: &[u8]) -> usize {
        let size = self.qh.max_packet_len().min(buffer.len());
        let written = self.buffers[0].volatile_write(&buffer[..size]);
        self.buffers[0].clean_invalidate_dcache(size);
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

    /// Schedule a transfer of `size` bytes from the endpoint buffer (legacy/EP0 path).
    ///
    /// Caller should check to see if there is an active transfer, or if the previous
    /// transfer resulted in an error or halt. Operates on slot 0.
    pub fn schedule_transfer(&mut self, usb: &ral::AnyUsbInstance, size: usize) {
        self.tds[0].set_terminate();
        self.tds[0].set_buffer(self.buffers[0].as_ptr_mut(), size);
        self.tds[0].set_interrupt_on_complete(true);
        self.tds[0].set_active();
        self.tds[0].clean_invalidate_dcache();

        self.qh.overlay_mut().set_next(&self.tds[0]);
        self.qh.overlay_mut().clear_status();
        self.qh.clean_invalidate_dcache();

        match self.address.direction() {
            UsbDirection::In => {
                ral::write_reg!(ral::usb, usb, ENDPTPRIME, PETB: 1 << self.address.index())
            }
            UsbDirection::Out => {
                ral::write_reg!(ral::usb, usb, ENDPTPRIME, PERB: 1 << self.address.index())
            }
        }
        while ral::read_reg!(ral::usb, usb, ENDPTPRIME) != 0 {}
    }

    /// Prime ONE dTD over `ptr[..size]` (Lever A big-dTD path, slot 0).
    ///
    /// The QH MPS stays 512; the controller auto-splits the dTD into MPS-sized
    /// USB packets. There are no cache operations here: the consuming firmware
    /// keeps the L1 D-cache OFF, so `clean_invalidate` is dead weight on this path.
    ///
    /// **Note**: `schedule_bulk` and the multi-TD ring (`schedule_next`/`complete`) are
    /// mutually exclusive per endpoint at runtime. `schedule_bulk` operates exclusively
    /// on slot 0 and does not interact with head/tail/in_flight ring bookkeeping.
    ///
    /// # Safety / ownership contract
    ///
    /// `ptr[..size]` must remain valid and unmodified until the transfer
    /// completes — i.e. until [`bulk_bytes_transferred`](Endpoint::bulk_bytes_transferred)
    /// is called (which the caller does only after confirming `is_primed` returned
    /// `false`).
    pub fn schedule_bulk(&mut self, usb: &ral::AnyUsbInstance, ptr: *mut u8, size: usize) {
        self.tds[0].set_terminate();
        self.tds[0].set_buffer(ptr, size);
        self.tds[0].set_interrupt_on_complete(true);
        self.tds[0].set_active();
        // D-cache is OFF in the consuming firmware: NO td.clean_invalidate_dcache /
        // buffer.clean_invalidate_dcache calls here (they would be dead weight).

        self.qh.overlay_mut().set_next(&self.tds[0]);
        self.qh.overlay_mut().clear_status();

        // Barrier: the dTD/overlay writes above land in (bufferable) SRAM, while
        // ENDPTPRIME is a Device-memory write that makes the controller (a separate
        // AHB bus master) fetch the dTD. ARM permits reordering a Normal-memory
        // store after a Device store, so without a DSB the controller can fetch a
        // stale dTD before our stores drain from the M7 store buffer. The legacy
        // `schedule_transfer` got this barrier for free inside `clean_invalidate_dcache`;
        // dropping the (D-cache-off) cache ops also dropped that DSB, so issue it
        // explicitly here.
        cortex_m::asm::dsb();

        match self.address.direction() {
            UsbDirection::In => {
                ral::write_reg!(ral::usb, usb, ENDPTPRIME, PETB: 1 << self.address.index())
            }
            UsbDirection::Out => {
                ral::write_reg!(ral::usb, usb, ENDPTPRIME, PERB: 1 << self.address.index())
            }
        }
        while ral::read_reg!(ral::usb, usb, ENDPTPRIME) != 0 {}
    }

    /// Returns the number of bytes moved by the most-recently-completed bulk dTD.
    ///
    /// Only call this after confirming that `is_primed` returned `false`
    /// (i.e. the transfer is complete). Reads from slot 0 (Lever A path).
    pub fn bulk_bytes_transferred(&self) -> usize {
        self.tds[0].bytes_transferred()
    }

    // -------------------------------------------------------------------------
    // Ring pump (multi-TD hot path, no cache ops)
    // -------------------------------------------------------------------------

    /// Prime the next free ring slot with `data` (IN) or as a receive buffer (OUT).
    ///
    /// For an IN endpoint, `data` contains bytes to send; for an OUT endpoint, `data`
    /// is ignored — the slot is primed with `max_packet_len` receive capacity.
    ///
    /// Returns the number of bytes accepted from `data` (IN), or 0 (OUT).
    ///
    /// # Panics
    ///
    /// Panics (debug) if called while `all_busy()` is true. In release mode the ring
    /// will silently corrupt — callers must check `!all_busy()` first.
    ///
    /// # No cache operations
    ///
    /// The consuming firmware keeps L1 D-cache OFF. This function contains no
    /// `clean_invalidate_dcache` calls.
    pub fn schedule_next(&mut self, usb: &ral::AnyUsbInstance, data: &[u8]) -> usize {
        debug_assert!(!self.all_busy(), "schedule_next called while ring full");

        let slot = self.head;
        let mps = self.qh.max_packet_len();

        // For IN: write data into the slot buffer and prime that many bytes.
        // For OUT: prime with max_packet_len receive capacity; data is irrelevant.
        let (td_size, written) = match self.address.direction() {
            UsbDirection::In => {
                let size = mps.min(data.len());
                self.buffers[slot].volatile_write(&data[..size]);
                (size, size)
            }
            UsbDirection::Out => (mps, 0),
        };

        // Program the TD for this slot.
        self.tds[slot].set_terminate();
        self.tds[slot].set_buffer(self.buffers[slot].as_ptr_mut(), td_size);
        self.tds[slot].set_interrupt_on_complete(true);
        self.tds[slot].set_active();
        // No cache ops — D-cache is OFF.

        let ep_index = self.address.index();

        if self.in_flight == 0 {
            // QH is idle — first dTD: write directly to overlay and prime.
            self.qh.overlay_mut().set_next(&self.tds[slot]);
            self.qh.overlay_mut().clear_status();
            // Barrier: flush the dTD + overlay stores (bufferable SRAM) before the
            // ENDPTPRIME Device-memory write hands the dTD to the controller (a
            // separate AHB master). Without it the controller can fetch a stale dTD
            // (e.g. a not-yet-terminated NEXT pointer → it follows garbage and halts
            // the queue, so ENDPTSTAT/ERBR never goes ready and the OUT transfer is
            // lost). The legacy `schedule_transfer` got this DSB for free from its
            // (D-cache-off, otherwise-dead) cache ops; the ring path must issue it.
            cortex_m::asm::dsb();
            match self.address.direction() {
                UsbDirection::In => {
                    ral::write_reg!(ral::usb, usb, ENDPTPRIME, PETB: 1 << ep_index)
                }
                UsbDirection::Out => {
                    ral::write_reg!(ral::usb, usb, ENDPTPRIME, PERB: 1 << ep_index)
                }
            }
            while ral::read_reg!(ral::usb, usb, ENDPTPRIME) != 0 {}
        } else {
            // QH is active — append to the live chain.
            // Step 1: link previous chain-tail's NEXT to the new TD.
            // Barrier first: the new dTD's payload (set_terminate/buffer/active above)
            // must be visible in SRAM before we publish the pointer to it, or the
            // controller can follow the link and fetch a half-written dTD.
            cortex_m::asm::dsb();
            let prev = (slot + self.depth() - 1) % self.depth();
            self.tds[prev].set_next(&self.tds[slot]);

            // Step 2: if ENDPTPRIME is already set for this EP, the controller will
            // pick up the new dTD automatically — we are done.
            // ENDPTPRIME layout: PETB (IN) = bits 16..23, PERB (OUT) = bits 0..7.
            let prime_bit: u32 = match self.address.direction() {
                UsbDirection::In => 1 << (ep_index + 16),
                UsbDirection::Out => 1 << ep_index,
            };
            let already_priming = ral::read_reg!(ral::usb, usb, ENDPTPRIME) & prime_bit != 0;

            if !already_priming {
                // Step 3: ATDTW tripwire — atomically check whether the EP is still
                // active while setting the ATDTW bit.
                //
                // ENDPTSTAT layout: ETBR (IN) = bits 16..23, ERBR (OUT) = bits 0..7.
                // Both use ep_index within their respective half-word.
                let stat_bit: u32 = match self.address.direction() {
                    UsbDirection::In => 1 << (ep_index + 16),
                    UsbDirection::Out => 1 << ep_index,
                };

                let mut ep_active;
                loop {
                    ral::modify_reg!(ral::usb, usb, USBCMD, ATDTW: 1);
                    // Read ENDPTSTAT while ATDTW is held.
                    let endptstat = ral::read_reg!(ral::usb, usb, ENDPTSTAT);
                    ep_active = endptstat & stat_bit != 0;
                    // Only trust the result if ATDTW is still 1 (no race).
                    if ral::read_reg!(ral::usb, usb, USBCMD, ATDTW == 1) {
                        break;
                    }
                }
                ral::modify_reg!(ral::usb, usb, USBCMD, ATDTW: 0);

                if !ep_active {
                    // EP went idle before the append landed: re-prime from this slot.
                    self.qh.overlay_mut().set_next(&self.tds[slot]);
                    self.qh.overlay_mut().clear_status();
                    // Flush overlay stores before the ENDPTPRIME Device write (see the
                    // first-prime branch for the full rationale).
                    cortex_m::asm::dsb();
                    match self.address.direction() {
                        UsbDirection::In => {
                            ral::write_reg!(ral::usb, usb, ENDPTPRIME, PETB: 1 << ep_index)
                        }
                        UsbDirection::Out => {
                            ral::write_reg!(ral::usb, usb, ENDPTPRIME, PERB: 1 << ep_index)
                        }
                    }
                    while ral::read_reg!(ral::usb, usb, ENDPTPRIME) != 0 {}
                }
                // If ep_active == true: the controller already sees the new TD via
                // the NEXT link we wrote at step 1. Nothing more to do.
            }
        }

        self.head = (slot + 1) % self.depth();
        self.in_flight += 1;

        written
    }

    /// Drain completed **IN** TDs from the ring tail.
    ///
    /// A TD is considered done when its ACTIVE bit is clear. Walks from `tail`
    /// forward until either the ring is empty or the next slot is still active.
    ///
    /// OUT endpoints are intentionally **not** drained here — their per-slot
    /// payloads are consumed one slot at a time by [`read_ring`](Endpoint::read_ring),
    /// which delivers each slot's bytes to the class exactly once. Draining OUT
    /// here (once per `poll`) collapsed every slot completed since the last poll
    /// into a single `last_*` snapshot, losing all but the newest slot's data and
    /// wedging the OUT ring during a multi-packet host→device data phase.
    ///
    /// Returns the number of slots freed (always 0 for an OUT endpoint).
    pub fn complete(&mut self, _usb: &ral::AnyUsbInstance) -> usize {
        if self.address.direction() == UsbDirection::Out {
            // OUT draining is owned by `read_ring` (per-slot delivery).
            return 0;
        }
        let mut freed = 0usize;
        while self.in_flight > 0 {
            let tail = self.tail;
            if self.tds[tail].status().contains(Status::ACTIVE) {
                // Still in flight — stop draining.
                break;
            }
            self.tail = (tail + 1) % self.depth();
            self.in_flight -= 1;
            freed += 1;
        }

        freed
    }

    /// Consume ONE completed OUT ring slot into `buf`, then re-prime that slot.
    ///
    /// This is the per-slot OUT delivery path. It inspects the oldest in-flight
    /// slot (`tail`): if its dTD is still ACTIVE the data has not arrived yet and
    /// it returns `None` (caller treats as `WouldBlock`). Otherwise it copies that
    /// slot's received bytes into `buf`, advances `tail`, decrements `in_flight`,
    /// and re-primes the freed slot so the ring stays full of receive capacity.
    ///
    /// Returns `Some(bytes_copied)` for a delivered slot, `None` when the tail slot
    /// is still in flight (ring momentarily empty of completed data).
    ///
    /// Gating on the tail TD's hardware ACTIVE bit (not a once-per-poll snapshot)
    /// lets the BOT read pump call this repeatedly within a single `poll` and
    /// drain every packet the controller has delivered, exactly once each.
    pub fn read_ring(&mut self, usb: &ral::AnyUsbInstance, buf: &mut [u8]) -> Option<usize> {
        debug_assert!(self.address.direction() == UsbDirection::Out);
        if self.in_flight == 0 {
            return None;
        }
        let tail = self.tail;
        if self.tds[tail].status().contains(Status::ACTIVE) {
            // Oldest slot not yet completed — nothing to deliver right now.
            return None;
        }
        let n = self.tds[tail].bytes_transferred().min(buf.len());
        let copied = self.buffers[tail].volatile_read(&mut buf[..n]);
        self.tail = (tail + 1) % self.depth();
        self.in_flight -= 1;
        // Re-prime the freed slot so the controller always has receive capacity.
        // `schedule_next` primes `self.head`, which after the decrement above has a
        // free slot available (`!all_busy`).
        if !self.all_busy() {
            self.schedule_next(usb, &[]);
        }
        Some(copied)
    }

    /// Reset ring bookkeeping and re-terminate every used TD.
    ///
    /// Called on (re)initialize after a bus reset (where the controller has
    /// flushed its primed TDs) and on unstall recovery, so the ring's
    /// `head`/`tail`/`in_flight` match the hardware's idle state.
    pub fn reset_ring(&mut self) {
        for td in self.tds.iter_mut().take(self.buffers.len()) {
            td.set_terminate();
            td.clear_status();
        }
        self.head = 0;
        self.tail = 0;
        self.in_flight = 0;
    }

    // -------------------------------------------------------------------------
    // Register helpers (unchanged)
    // -------------------------------------------------------------------------

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
    // Test helpers (compiled out in production builds)
    // -------------------------------------------------------------------------

    /// Returns the number of active ring slots. Test helper.
    #[cfg(test)]
    pub(crate) fn ring_depth(&self) -> usize {
        self.depth()
    }

    /// Advance bookkeeping only, without touching USB registers.
    ///
    /// Used by unit tests to prime ring slots without a real USB peripheral.
    /// Sets the TD ACTIVE bit directly and advances head/in_flight.
    #[cfg(test)]
    pub(crate) fn test_prime_slot(&mut self) {
        debug_assert!(!self.all_busy());
        let slot = self.head;
        self.tds[slot].set_terminate();
        self.tds[slot].set_buffer(self.buffers[slot].as_ptr_mut(), self.buffers[slot].len());
        self.tds[slot].set_interrupt_on_complete(true);
        self.tds[slot].set_active();
        self.head = (slot + 1) % self.depth();
        self.in_flight += 1;
    }

    /// Mark the TD at `slot` as completed (clear ACTIVE) for bookkeeping tests.
    #[cfg(test)]
    pub(crate) fn test_complete_slot(&mut self, slot: usize) {
        self.tds[slot].clear_status();
    }

    /// Expose `in_flight` for assertions in tests.
    #[cfg(test)]
    pub(crate) fn in_flight(&self) -> u32 {
        self.in_flight
    }

    /// Expose `tail` for assertions in tests.
    #[cfg(test)]
    pub(crate) fn tail(&self) -> usize {
        self.tail
    }

    /// Expose `head` for assertions in tests.
    #[cfg(test)]
    pub(crate) fn head(&self) -> usize {
        self.head
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
// Unit tests (bookkeeping only — no USB register access)
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Fake USB instance for tests that need &ral::AnyUsbInstance but never
    // dereference it (complete() currently ignores the argument entirely).
    // -----------------------------------------------------------------------

    /// Return a reference to a zeroed, static register-block stand-in.
    ///
    /// `complete()` accepts `&ral::AnyUsbInstance` but never dereferences it,
    /// so a dangling-but-non-null pointer is safe here.  We use `static mut`
    /// (bypasses the `Sync` requirement on `RegisterBlock`) and obtain a raw
    /// const pointer via `addr_of!` to satisfy Edition 2024's ban on
    /// `&*static_mut`.
    fn fake_usb() -> &'static imxrt_ral::usb::Instance<255> {
        static mut FAKE_USB_BLK: imxrt_ral::usb::RegisterBlock =
            // SAFETY: all-zero is a valid bit pattern for a register block used
            // only as an address; we never read or write through this in tests.
            unsafe { core::mem::zeroed() };
        // SAFETY: addr_of! does not form a reference to the static; the cast is
        // valid because Instance<N> is repr(transparent) over the register block.
        unsafe { &*(core::ptr::addr_of!(FAKE_USB_BLK) as *const imxrt_ral::usb::Instance<255>) }
    }

    // -----------------------------------------------------------------------
    // Helpers to construct an Endpoint without a USB peripheral
    // -----------------------------------------------------------------------

    /// Stack-allocate backing store and build an Endpoint with `depth` slots,
    /// binding it to `$ep_name` in the caller's scope.
    ///
    /// The macro takes an extra first argument for the output variable name to
    /// work around Rust macro hygiene (variables defined inside a macro are not
    /// visible in the caller).
    ///
    /// Edition 2024 note: we use `core::ptr::addr_of_mut!` to obtain a raw
    /// pointer to the static backing, then dereference it inside `unsafe {}`,
    /// avoiding the forbidden `&mut STATIC_MUT` form.
    macro_rules! make_endpoint {
        ($ep_name:ident, $name:ident, $depth:expr, $mps:expr, $dir:expr, $kind:expr) => {
            // Backing for the TD ring. Each Td is 32 bytes; repr(align(32))
            // on TdList ensures proper alignment in production. On x86_64 the
            // linker also satisfies align(32) for statics, so tests are valid.
            static mut $name: (
                crate::qh::Qh,
                [crate::td::Td; crate::state::RING_DEPTH],
                [[u8; $mps]; crate::state::RING_DEPTH],
            ) = (
                crate::qh::Qh::new(),
                [const { crate::td::Td::new() }; crate::state::RING_DEPTH],
                [[0u8; $mps]; crate::state::RING_DEPTH],
            );

            // Edition 2024: use addr_of_mut! to get a raw pointer, then
            // dereference individual fields — never form &mut STATIC directly.
            let ep_qh_ref: &'static mut crate::qh::Qh =
                unsafe { &mut (*core::ptr::addr_of_mut!($name)).0 };
            let ep_tds_slice: &'static mut [crate::td::Td] =
                unsafe { &mut (*core::ptr::addr_of_mut!($name)).1 };

            let mut ep_bufs = heapless::Vec::<crate::buffer::Buffer, RING_DEPTH>::new();
            for i in 0..$depth {
                // SAFETY: each sub-array lives for 'static (it's in a static).
                // addr_of_mut! avoids the forbidden &mut-of-static form.
                let raw_ptr: *mut u8 =
                    unsafe { (*core::ptr::addr_of_mut!($name)).2[i].as_mut_ptr() };
                let mut sub_alloc = unsafe {
                    crate::buffer::Allocator::from_buffer(core::slice::from_raw_parts_mut(
                        raw_ptr, $mps,
                    ))
                };
                let _ = ep_bufs.push(sub_alloc.allocate($mps).unwrap());
            }

            let mut $ep_name = crate::endpoint::Endpoint::new(
                usb_device::endpoint::EndpointAddress::from_parts(1, $dir),
                ep_qh_ref,
                ep_tds_slice,
                ep_bufs,
                $kind,
            );
        };
    }

    /// Fill the ring RING_DEPTH times and verify `all_busy()`, then clear two
    /// oldest TDs and verify `complete()` returns 2, tail advanced, in_flight correct.
    #[test]
    fn ring_prime_until_full_then_drain() {
        make_endpoint!(
            ep,
            BACKING,
            RING_DEPTH,
            64,
            UsbDirection::In,
            EndpointType::Bulk
        );

        // Ring must start empty.
        assert!(!ep.all_busy());
        assert_eq!(ep.in_flight(), 0);

        // Prime all slots via bookkeeping helper.
        for _ in 0..RING_DEPTH {
            assert!(!ep.all_busy());
            ep.test_prime_slot();
        }
        assert!(ep.all_busy(), "ring should be full after RING_DEPTH primes");
        assert_eq!(ep.in_flight(), RING_DEPTH as u32);
        assert_eq!(ep.head(), 0, "head wraps back to 0 after filling");

        // Mark the two oldest TDs (slots 0 and 1) as completed.
        ep.test_complete_slot(0);
        ep.test_complete_slot(1);

        let freed = ep.complete(fake_usb());
        assert_eq!(freed, 2, "complete() must drain the 2 cleared TDs");
        assert_eq!(ep.tail(), 2, "tail must advance by 2");
        assert_eq!(
            ep.in_flight(),
            (RING_DEPTH - 2) as u32,
            "in_flight must decrease by 2"
        );
        assert_eq!(ep.head(), 0, "head unchanged by complete()");

        // The next slot to prime is index 0 (the first freed slot).
        assert!(!ep.all_busy());
    }

    /// `reset_ring` returns the bookkeeping to the empty state regardless of how
    /// many slots were in flight — the coherence restore a bus reset relies on.
    #[test]
    fn reset_ring_clears_bookkeeping() {
        make_endpoint!(
            ep,
            BACKING_RST,
            RING_DEPTH,
            64,
            UsbDirection::In,
            EndpointType::Bulk
        );

        // Drive the ring into a partially-filled state: prime 3, drain 1.
        for _ in 0..3 {
            ep.test_prime_slot();
        }
        ep.test_complete_slot(0);
        assert_eq!(ep.complete(fake_usb()), 1);
        assert_eq!(ep.in_flight(), 2);
        assert_ne!(ep.head(), 0);

        ep.reset_ring();

        assert_eq!(ep.in_flight(), 0);
        assert_eq!(ep.head(), 0);
        assert_eq!(ep.tail(), 0);
        assert!(!ep.all_busy());
    }

    /// A depth-1 ring behaves like the legacy single-TD path.
    #[test]
    fn depth1_equals_legacy() {
        make_endpoint!(ep, BACKING1, 1, 64, UsbDirection::In, EndpointType::Control);

        assert!(!ep.all_busy());
        assert_eq!(ep.in_flight(), 0);

        ep.test_prime_slot();
        assert!(ep.all_busy(), "depth-1 ring busy after one prime");
        assert_eq!(ep.in_flight(), 1);

        // TD still ACTIVE — complete() must return 0.
        let freed = ep.complete(fake_usb());
        assert_eq!(freed, 0, "complete() must return 0 while ACTIVE");
        assert_eq!(ep.in_flight(), 1);

        // Mark complete.
        ep.test_complete_slot(0);
        let freed = ep.complete(fake_usb());
        assert_eq!(freed, 1, "complete() must return 1 after clearing ACTIVE");
        assert_eq!(ep.in_flight(), 0);
        assert!(!ep.all_busy());
    }
}
