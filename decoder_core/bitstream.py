"""Small Annex B writer: High profile, I_PCM inputs, weighted P/B predictions.

Each prediction macroblock has its own slice, so motion-vector predictors are zero.
Prediction pictures contain no transform coefficients or residual samples.
"""

from dataclasses import dataclass, replace


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
        self.align()
        return bytes(sum(self.bits[i + j] << (7 - j) for j in range(8))
                     for i in range(0, len(self.bits), 8))


def nal(kind, bits, reference=3):
    escaped = bytearray()
    zeros = 0
    for value in bits.finish():
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


class Writer:
    MAX_REFS = 16

    def __init__(self, width, height, fps=30, motion_bits=4):
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
        if self.mb_width * self.mb_height * fps > 1800:
            raise ValueError("One prediction slice per macroblock exceeds level 3.1 slice rate; reduce board size or fps")
        self.fps = fps
        if not 0 <= motion_bits <= 16:
            raise ValueError("Motion-vector bound must use 0..16 bits")
        self.motion_bits = motion_bits

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
        b = Bits()
        sources = operation.sources if operation else ()
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
                distance = frame - source.frame
                if not 1 <= distance <= self.MAX_REFS:
                    raise ValueError(f"Reference outside DPB: {distance}")
                b.u(1)  # ref_pic_list_modification_flag_lX
                b.ue(0)  # subtract from CurrPicNum
                b.ue(distance - 1)
                b.ue(3)  # end list modification
            b.ue(operation.denominator)
            b.ue(0)  # chroma_log2_weight_denom
            for source in sources:
                if not -128 <= source.weight <= 127 or not -128 <= source.offset <= 127:
                    raise ValueError("Weight or offset outside signed 8-bit range")
                b.u(1)
                b.se(source.weight)
                b.se(source.offset)
                b.u(0)  # chroma weights default: keep neutral 128
        if idr:
            b.u(0)  # no_output_of_prior_pics_flag
            b.u(0)  # long_term_reference_flag
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
        for my in range(self.mb_height):
            for mx in range(self.mb_width):
                b.ue(25)  # I_PCM
                b.align()
                for y in range(my * 16, my * 16 + 16):
                    for x in range(mx * 16, mx * 16 + 16):
                        inside = 16 <= x < self.width - 16 and 16 <= y < self.height - 16
                        b.u(board[y - 16][x - 16] if inside else 0, 8)
                for _ in range(128):
                    b.u(128, 8)
        out = (self.parameter_sets() + nal(5, b)) if frame == 0 else nal(1, b)
        # High profile MinCR=4; initial AU bound, Annex A.3.3(i).
        # Exactly three NALs (SPS, PPS, IDR); four-byte start codes don't count.
        initial_limit = 96 * max(self.mb_width * self.mb_height * 172, 108000) // 172
        if frame == 0 and len(out) - 12 > initial_limit:
            raise ValueError("PCM seed exceeds High level 3.1 initial access-unit size limit")
        return out

    def prediction(self, frame, operation):
        def motion(source):
            components = (source.dx, source.dy, source.qx, source.qy)
            if any(not isinstance(value, int) or isinstance(value, bool) for value in components):
                raise ValueError("Motion-vector components must be integers")
            vector = (source.dx * 4 + source.qx, source.dy * 4 + source.qy)
            if any(not -(1 << self.motion_bits) <= component < (1 << self.motion_bits)
                   for component in vector):
                raise ValueError("Motion vector exceeds declared SPS bound")
            return vector

        for source in operation.sources:
            motion(source)
        offsets = operation.block_offsets
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
                motion(replace(source, qx=source.qx + offset[0], qy=source.qy + offset[1]))
        out = bytearray()
        for my in range(self.mb_height):
            for mx in range(self.mb_width):
                border = mx in (0, self.mb_width - 1) or my in (0, self.mb_height - 1)
                op = operation
                if border:
                    op = replace(operation, sources=tuple(replace(s, weight=0, offset=0,
                                 dx=0, dy=0, qx=0, qy=0) for s in operation.sources))
                elif offsets:
                    index = (my - 1) * (self.mb_width - 2) + mx - 1
                    qx, qy = offsets[index]
                    op = replace(operation, sources=(replace(operation.sources[0],
                                 qx=operation.sources[0].qx + qx, qy=operation.sources[0].qy + qy),))
                b = self.header(my * self.mb_width + mx, frame, op)
                b.ue(0)  # mb_skip_run
                b.ue(0 if len(op.sources) == 1 else 3)  # P_L0_16x16 / B_Bi_16x16
                for source in op.sources:
                    # Slice boundaries make every neighboring predictor unavailable.
                    qx, qy = motion(source)
                    b.se(qx)
                    b.se(qy)
                b.ue(0)  # coded_block_pattern: no residual
                out.extend(nal(1, b))
        return bytes(out)
