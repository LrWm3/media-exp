"""Small Annex B writer: High profile, I_PCM inputs, weighted P/B predictions.

Prediction slices cover individual macroblocks or compatible horizontal runs.
Prediction pictures contain no transform coefficients or residual samples.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType


class Bits:
    def __init__(self):
        self.bits = []

    def u(self, value, width=1):
        if not 0 <= value < 1 << width:
            raise ValueError((value, width))
        self.bits.extend((value >> i) & 1 for i in reversed(range(width)))

    def ue(self, value):
        if value < 0:
            raise ValueError(value)
        code = value + 1
        self.u(0, code.bit_length() - 1)
        self.u(code, code.bit_length())

    def se(self, value):
        self.ue(-2 * value if value <= 0 else 2 * value - 1)

    def align(self):
        while len(self.bits) % 8:
            self.u(0)

    def finish(self):
        self.u(1)  # rbsp_stop_one_bit
        return self.aligned_bytes()

    def aligned_bytes(self):
        """Zero-pad to a byte boundary without adding an RBSP stop bit."""
        self.align()
        return bytes(sum(self.bits[i + j] << (7 - j) for j in range(8))
                     for i in range(0, len(self.bits), 8))


def nal(kind, bits, reference=3):
    return nal_from_rbsp(kind, bits.finish(), reference)


def nal_from_rbsp(kind, rbsp, reference=3):
    escaped = bytearray()
    zeros = 0
    for value in rbsp:
        if zeros == 2 and value <= 3:
            escaped.append(3)
            zeros = 0
        escaped.append(value)
        zeros = zeros + 1 if value == 0 else 0
    return b"\x00\x00\x00\x01" + bytes([(reference << 5) | kind]) + escaped


@dataclass(frozen=True)
class Source:
    frame: int
    weight: int = 1
    offset: int = 0
    dx: int = 0
    dy: int = 0
    # Additional quarter-pixel displacements: MV = (4*dx + qx, 4*dy + qy).
    qx: int = 0
    qy: int = 0


@dataclass(frozen=True)
class Operation:
    label: str
    sources: tuple[Source, ...]
    denominator: int = 0
    # Extra quarter-pixel displacements for active 16x16 blocks, in row order.
    block_offsets: tuple[tuple[int, int], ...] = ()
    # Complete predictions for active 16x16 blocks, in row order. Children use
    # the parent's source count and cannot carry either kind of block override.
    block_operations: tuple["Operation", ...] = ()


class Writer:
    MAX_REFS = 16

    def __init__(self, width, height, fps=30, motion_bits=4,
                 reference_slots=None, slice_mode="macroblock"):
        """Create a writer with an optional immutable physical-frame slot plan.

        A plan assigns every frame, starting at zero, a long-term slot in 0..15.
        Frame zero occupies slot zero. Reusing a slot replaces its old picture
        after decoding the current picture, so the current picture may still
        read that old occupant. Without a plan, the legacy sliding window applies.
        Encoding never changes reference state; repeated or out-of-order calls
        produce the same bytes for the same arguments.
        """
        # A 16-pixel guard on every side enforces a finite, dead boundary.
        if width < 16 or height < 16 or width % 16 or height % 16:
            raise ValueError("Board dimensions must be positive multiples of 16")
        self.width, self.height = width + 32, height + 32
        self.mb_width, self.mb_height = self.width // 16, self.height // 16
        if max(self.mb_width, self.mb_height) > 169:
            raise ValueError("Board exceeds level 3.1 per-axis dimension limit")
        if self.mb_width * self.mb_height * self.MAX_REFS > 18000:
            raise ValueError("Board exceeds level 3.1 decoded-picture buffer limit")
        if not 1 <= fps <= 60:
            raise ValueError("fps must be between 1 and 60")
        if slice_mode not in ("macroblock", "row"):
            raise ValueError("slice_mode must be 'macroblock' or 'row'")
        self.slice_mode = slice_mode
        slices = (self.mb_width * self.mb_height if slice_mode == "macroblock"
                  else 3 * (self.mb_height - 2) + 2)
        if slices * fps > 1800:
            raise ValueError("Prediction slices exceed level 3.1 slice rate; reduce board size or fps")
        self.fps = fps
        if not 0 <= motion_bits <= 16:
            raise ValueError("Motion-vector bound must use 0..16 bits")
        self.motion_bits = motion_bits
        self.reference_slots = None
        self._reference_expiry = {}
        if reference_slots is not None:
            if not isinstance(reference_slots, Mapping):
                raise ValueError("Reference slots must map every physical frame to a slot")
            slots = dict(reference_slots)
            if (not slots or any(type(frame) is not int or frame < 0 for frame in slots)
                    or sorted(slots) != list(range(len(slots)))):
                raise ValueError("Reference slots must cover consecutive frames starting at zero")
            if any(type(slot) is not int or not 0 <= slot < self.MAX_REFS for slot in slots.values()):
                raise ValueError("Reference slots must be integers from 0 through 15")
            if slots[0] != 0:
                raise ValueError("IDR frame zero must occupy reference slot zero")
            self.reference_slots = MappingProxyType(slots)
            occupants = {}
            for frame in range(len(slots)):
                slot = slots[frame]
                if slot in occupants:
                    self._reference_expiry[occupants[slot]] = frame
                occupants[slot] = frame

    def _validate_frame(self, frame):
        if type(frame) is not int or frame < 0:
            raise ValueError("Picture frame must be a nonnegative integer")
        if self.reference_slots is not None and frame not in self.reference_slots:
            raise ValueError(f"Picture frame {frame} missing from reference slot plan")

    def _reference_index(self, frame, source):
        if type(source.frame) is not int or not 0 <= source.frame < frame:
            raise ValueError("Reference frame must precede the current picture")
        if self.reference_slots is None:
            distance = frame - source.frame
            if distance > self.MAX_REFS:
                raise ValueError(f"Reference outside DPB: {distance}")
            return distance - 1
        if self._reference_expiry.get(source.frame, frame) < frame:
            raise ValueError(f"Reference frame {source.frame} no longer resident in its slot")
        return self.reference_slots[source.frame]

    @staticmethod
    def _validate_operation(operation):
        if len(operation.sources) not in (1, 2):
            raise ValueError("Prediction requires one or two sources")
        if type(operation.denominator) is not int or not 0 <= operation.denominator <= 7:
            raise ValueError("Weight denominator must be an integer from 0 through 7")
        for source in operation.sources:
            if (type(source.weight) is not int or type(source.offset) is not int
                    or not -128 <= source.weight <= 127 or not -128 <= source.offset <= 127):
                raise ValueError("Weight or offset outside signed 8-bit range")

    def parameter_sets(self):
        b = Bits()
        b.u(100, 8)  # High profile permits zero-valued I_PCM samples
        b.u(0, 8)
        b.u(31, 8)  # level 3.1
        b.ue(0)
        b.ue(1)  # chroma_format_idc: 4:2:0
        b.ue(0)  # bit_depth_luma_minus8
        b.ue(0)  # bit_depth_chroma_minus8
        b.u(0)  # qpprime_y_zero_transform_bypass_flag
        b.u(0)  # seq_scaling_matrix_present_flag
        b.ue(12)  # 16-bit frame_num
        b.ue(0)  # pic_order_cnt_type
        b.ue(12)  # 16-bit POC LSB
        b.ue(self.MAX_REFS)
        b.u(0)  # no gaps in frame_num
        b.ue(self.mb_width - 1)
        b.ue(self.mb_height - 1)
        b.u(1)  # frame_mbs_only_flag
        b.u(1)  # direct_8x8_inference_flag
        b.u(0)  # no cropping
        b.u(1)  # VUI
        b.u(0)  # aspect_ratio_info_present_flag
        b.u(0)  # overscan_info_present_flag
        b.u(1)  # video_signal_type_present_flag
        b.u(5, 3)  # unspecified video format
        b.u(1)  # full-range luma
        b.u(0)  # colour_description_present_flag
        b.u(0)  # chroma_loc_info_present_flag
        b.u(1)  # timing_info_present_flag
        b.u(1, 32)
        b.u(self.fps * 2, 32)
        b.u(1)  # fixed_frame_rate_flag
        b.u(0)  # nal_hrd_parameters_present_flag
        b.u(0)  # vcl_hrd_parameters_present_flag
        b.u(0)  # pic_struct_present_flag
        b.u(1)  # bitstream_restriction_flag
        b.u(1)  # motion_vectors_over_pic_boundaries_flag
        b.ue(0)  # no VCL byte bound: PCM seed includes raw samples plus headers
        b.ue(1)  # max_bits_per_mb_denom
        b.ue(self.motion_bits)  # declared horizontal/vertical quarter-sample bounds
        b.ue(self.motion_bits)
        b.ue(0)  # max_num_reorder_frames
        b.ue(self.MAX_REFS)
        sps = nal(7, b)
        b = Bits()
        b.ue(0)  # PPS id
        b.ue(0)  # SPS id
        b.u(0)  # CAVLC
        b.u(0)  # bottom_field_pic_order_in_frame_present_flag
        b.ue(0)  # num_slice_groups_minus1
        b.ue(0)  # num_ref_idx_l0_default_active_minus1
        b.ue(0)  # num_ref_idx_l1_default_active_minus1
        b.u(1)  # weighted_pred_flag
        b.u(1, 2)  # explicit weighted biprediction
        b.se(0)  # pic_init_qp_minus26
        b.se(0)  # pic_init_qs_minus26
        b.se(0)  # chroma_qp_index_offset
        b.u(1)  # deblocking_filter_control_present_flag
        b.u(0)  # constrained_intra_pred_flag
        b.u(0)  # redundant_pic_cnt_present_flag
        return sps + nal(8, b)

    def header(self, address, frame, operation=None):
        self._validate_frame(frame)
        b = Bits()
        sources = operation.sources if operation else ()
        if operation is not None:
            self._validate_operation(operation)
        b.ue(address)
        b.ue({0: 2, 1: 0, 2: 1}[len(sources)])  # I, P, B
        b.ue(0)
        b.u(frame % 65536, 16)
        idr = not sources and frame == 0
        if idr:
            b.ue(0)  # idr_pic_id
        b.u((2 * frame) % 65536, 16)
        if len(sources) == 2:
            b.u(1)  # direct_spatial_mv_pred_flag (direct mode unused)
        if sources:
            b.u(1)  # num_ref_idx_active_override_flag
            for _ in sources:
                b.ue(0)  # one active entry per list
            for source in sources:
                b.u(1)  # ref_pic_list_modification_flag_lX
                b.ue(0 if self.reference_slots is None else 2)
                # abs_diff_pic_num_minus1, or long_term_pic_num for explicit slots.
                b.ue(self._reference_index(frame, source))
                b.ue(3)  # end list modification
            b.ue(operation.denominator)
            b.ue(0)  # chroma_log2_weight_denom
            for source in sources:
                b.u(1)
                b.se(source.weight)
                b.se(source.offset)
                b.u(0)  # chroma weights default: keep neutral 128
        if idr:
            b.u(0)  # no_output_of_prior_pics_flag
            b.u(int(self.reference_slots is not None))  # long_term_reference_flag
        elif self.reference_slots is not None:
            b.u(1)  # adaptive_ref_pic_marking_mode_flag
            b.ue(4)  # set maximum long-term index
            b.ue(self.MAX_REFS)  # max_long_term_frame_idx_plus1
            b.ue(6)  # assign this picture, replacing its slot's previous occupant
            b.ue(self.reference_slots[frame])
            b.ue(0)  # end memory-management commands
        else:
            b.u(0)  # adaptive_ref_pic_marking_mode_flag: sliding window
        b.se(0)  # slice_qp_delta
        b.ue(1)  # disable_deblocking_filter_idc
        return b

    def seed(self, board):
        if any(value not in (0, 1) for row in board for value in row):
            raise ValueError("Seed cells must be zero or one")
        return self.seed_luma([[255 * value for value in row] for row in board])

    def seed_luma(self, board):
        """Store arbitrary 8-bit initial conditions, with a permanent zero guard."""
        return self.input_luma(board, 0)

    def input_luma(self, board, frame):
        """Load a u8 input; only frame zero resets the reference-picture buffer."""
        if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
            raise ValueError("Input frame must be a nonnegative integer")
        if len(board) != self.height - 32 or any(len(row) != self.width - 32 for row in board):
            raise ValueError("Seed dimensions do not match board")
        if any(not isinstance(value, int) or not 0 <= value <= 255 for row in board for value in row):
            raise ValueError("Seed luma samples must be integers from 0 through 255")
        # One I_PCM slice; later non-IDR inputs retain earlier reference pictures.
        b = self.header(0, frame)
        b.ue(25)  # First I_PCM type shares the slice header's partial byte.
        rbsp = bytearray(b.aligned_bytes())
        rows = [bytes(row) for row in board]
        chroma = bytes([128]) * 128
        guard = bytes(256) + chroma
        for my in range(self.mb_height):
            for mx in range(self.mb_width):
                if my or mx:
                    rbsp.extend(b"\x0d\x00")  # ue(25), then seven pcm_alignment_zero_bits.
                if mx in (0, self.mb_width - 1) or my in (0, self.mb_height - 1):
                    rbsp.extend(guard)
                else:
                    x = (mx - 1) * 16
                    for y in range((my - 1) * 16, my * 16):
                        rbsp.extend(rows[y][x:x + 16])
                    rbsp.extend(chroma)
        rbsp.append(128)  # rbsp_stop_one_bit, then seven zero alignment bits.
        out = (self.parameter_sets() + nal_from_rbsp(5, rbsp)) if frame == 0 else nal_from_rbsp(1, rbsp)
        # High profile MinCR=4; initial AU bound, Annex A.3.3(i).
        # Exactly three NALs (SPS, PPS, IDR); four-byte start codes don't count.
        initial_limit = 96 * max(self.mb_width * self.mb_height * 172, 108000) // 172
        if frame == 0 and len(out) - 12 > initial_limit:
            raise ValueError("PCM seed exceeds High level 3.1 initial access-unit size limit")
        return out

    def _motion(self, source):
        components = (source.dx, source.dy, source.qx, source.qy)
        if any(type(value) is not int for value in components):
            raise ValueError("Motion-vector components must be integers")
        vector = (source.dx * 4 + source.qx, source.dy * 4 + source.qy)
        if any(not -(1 << self.motion_bits) <= component < (1 << self.motion_bits)
               for component in vector):
            raise ValueError("Motion vector exceeds declared SPS bound")
        return vector

    def _prediction_runs(self, operation):
        self._validate_operation(operation)
        for source in operation.sources:
            self._motion(source)
        offsets = operation.block_offsets
        blocks = operation.block_operations
        if blocks:
            if offsets:
                raise ValueError("Block operations and block offsets are mutually exclusive")
            if len(blocks) != (self.mb_width - 2) * (self.mb_height - 2):
                raise ValueError("Block operations must cover every active macroblock")
            for block in blocks:
                if not isinstance(block, Operation):
                    raise ValueError("Block operations must contain Operation values")
                if block.block_operations or block.block_offsets:
                    raise ValueError("Block operations cannot contain nested block overrides")
                self._validate_operation(block)
                if len(block.sources) != len(operation.sources):
                    raise ValueError("Block operations must have the parent's source count")
                for source in block.sources:
                    self._motion(source)
        if offsets:
            if len(operation.sources) != 1:
                raise ValueError("Block offsets require exactly one source")
            if len(offsets) != (self.mb_width - 2) * (self.mb_height - 2):
                raise ValueError("Block offsets must cover every active macroblock")
            source = operation.sources[0]
            for offset in offsets:
                if (not isinstance(offset, (tuple, list)) or len(offset) != 2 or
                        any(not isinstance(value, int) or isinstance(value, bool) for value in offset)):
                    raise ValueError("Block offsets must be integer quarter-pixel pairs")
                self._motion(replace(source, qx=source.qx + offset[0], qy=source.qy + offset[1]))
        guard = replace(operation, sources=tuple(replace(s, weight=0, offset=0,
                        dx=0, dy=0, qx=0, qy=0) for s in operation.sources),
                        block_offsets=(), block_operations=())
        for my in range(self.mb_height):
            run, previous_key = [], None
            for mx in range(self.mb_width):
                border = mx in (0, self.mb_width - 1) or my in (0, self.mb_height - 1)
                op = operation
                if border:
                    op = guard
                elif blocks:
                    index = (my - 1) * (self.mb_width - 2) + mx - 1
                    op = blocks[index]
                elif offsets:
                    index = (my - 1) * (self.mb_width - 2) + mx - 1
                    qx, qy = offsets[index]
                    op = replace(operation, sources=(replace(operation.sources[0],
                                 qx=operation.sources[0].qx + qx, qy=operation.sources[0].qy + qy),))
                # Motion belongs to macroblocks; only weights/references/denom
                # belong to the slice header and therefore split horizontal runs.
                key = (op.denominator, tuple((s.frame, s.weight, s.offset) for s in op.sources))
                if run and (self.slice_mode == "macroblock" or key != previous_key):
                    yield run
                    run = []
                run.append((my * self.mb_width + mx, op))
                previous_key = key
            yield run

    def slice_count(self, operation):
        """Return prediction VCL slice count without encoding or changing state."""
        return sum(1 for _ in self._prediction_runs(operation))

    def prediction(self, frame, operation):
        runs = self._prediction_runs(operation)
        if operation.block_operations:
            # Per-block headers can split rows more often than the uniform
            # prediction bound checked by the constructor. Check before writing.
            runs = list(runs)
            if len(runs) * self.fps > 1800:
                raise ValueError("Prediction slices exceed level 3.1 slice rate; reduce board size or fps")
            self._validate_frame(frame)
            for block in operation.block_operations:
                for source in block.sources:
                    # A matching run key does not establish valid frame types:
                    # True and 1.0 compare equal to frame 1 in Python.
                    self._reference_index(frame, source)
        out = bytearray()
        for run in runs:
            b = self.header(run[0][0], frame, run[0][1])
            previous = [(0, 0)] * len(operation.sources)
            for _, op in run:
                b.ue(0)  # mb_skip_run
                b.ue(0 if len(op.sources) == 1 else 3)  # P_L0_16x16 / B_Bi_16x16
                for index, source in enumerate(op.sources):
                    # Runs never cross a row. Only the left neighbor is available;
                    # its vector predicts this list's 16x16 vector (H.264 8.4.1.3).
                    qx, qy = self._motion(source)
                    px, py = previous[index]
                    b.se(qx - px)
                    b.se(qy - py)
                    previous[index] = (qx, qy)
                b.ue(0)  # coded_block_pattern: no residual
            out.extend(nal(1, b))
        return bytes(out)
