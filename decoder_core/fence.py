"""A tiny discarded IDR picture for completing resident decoder requests."""

from .bitstream import nal


def neutral_idr(writer, *, idr_pic_id=1):
    """Encode constant 128 I420 using intra DC prediction and zero residual.

    Each I_16x16_2_0_0 macroblock predicts DC luma/chroma from constant neighbors;
    unavailable picture edges use 128. The luma DC block has TotalCoeff=0. This
    picture is a runtime delimiter, never a graph result or an input reference
    for the next job. Unlike computation pictures, its guard is also 128.
    """
    # Consecutive IDR access units must have different idr_pic_id values
    # (H.264 7.4.3). Ordinary compiler inputs use zero; fences default to one.
    bits = writer.header(0, 0, idr_pic_id=idr_pic_id)
    for _ in range(writer.mb_width * writer.mb_height):
        bits.ue(3)   # I_16x16_2_0_0: DC, no coded luma AC or chroma residual.
        bits.ue(0)   # intra_chroma_pred_mode: DC.
        bits.se(0)   # mb_qp_delta is present for every I_16x16 macroblock.
        bits.u(1)    # coeff_token: TotalCoeff=0, nC=0.
    return writer._checked_access_unit(0, writer.parameter_sets() + nal(5, bits), "Resident fence")
