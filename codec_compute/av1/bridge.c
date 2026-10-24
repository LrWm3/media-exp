/* Deterministic AV1 serialization bridge. Built only with our pinned patch. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "aom/aom_encoder.h"
#include "aom/aomcx.h"

int codec_compute_mode = 0, codec_compute_dx = 0, codec_compute_dy = 0;
static void fail(const char *s) { fprintf(stderr, "%s\n", s); exit(1); }
static void check(aom_codec_ctx_t *ctx, aom_codec_err_t err) {
  if (err != AOM_CODEC_OK) {
    fprintf(stderr, "libaom: %s (%s)\n", aom_codec_err_to_string(err),
            aom_codec_error_detail(ctx) ? aom_codec_error_detail(ctx) : "");
    exit(1);
  }
}
static uint32_t read32(void) {
  unsigned char b[4]; if (fread(b, 1, 4, stdin) != 4) fail("short request");
  return (uint32_t)b[0] | (uint32_t)b[1]<<8 | (uint32_t)b[2]<<16 | (uint32_t)b[3]<<24;
}
static void write32(uint32_t n) {
  for (int i=0; i<4; ++i) putchar((n >> (8*i)) & 255);
}
#define CTRL(id, value) check(&ctx, aom_codec_control(&ctx, id, value))
int main(void) {
  const unsigned w=read32(), h=read32(), count=read32();
  if (!w || !h || w%16 || h%16 || w>4096 || h>4096 || !count) fail("invalid geometry/count");
  aom_codec_ctx_t ctx;
  aom_codec_enc_cfg_t cfg;
  check(NULL, aom_codec_enc_config_default(aom_codec_av1_cx(), &cfg, AOM_USAGE_REALTIME));
  cfg.g_w=w; cfg.g_h=h; cfg.g_threads=1;
  cfg.g_timebase.num=1; cfg.g_timebase.den=30;
  cfg.g_lag_in_frames=0; cfg.g_error_resilient=0;
  cfg.rc_end_usage=AOM_Q; cfg.rc_min_quantizer=0; cfg.rc_max_quantizer=0;
  cfg.kf_mode=AOM_KF_DISABLED; cfg.kf_max_dist=0;
  cfg.monochrome=1;
  check(&ctx, aom_codec_enc_init(&ctx, aom_codec_av1_cx(), &cfg, 0));
  CTRL(AOME_SET_CPUUSED, 8);
  CTRL(AV1E_SET_LOSSLESS, 1);
  CTRL(AV1E_SET_CDF_UPDATE_MODE, 0);
  CTRL(AV1E_SET_ENABLE_CDEF, 0);
  CTRL(AV1E_SET_ENABLE_RESTORATION, 0);
  CTRL(AV1E_SET_LOOPFILTER_CONTROL, 0);
  CTRL(AV1E_SET_ENABLE_ORDER_HINT, 0);
  CTRL(AV1E_SET_ENABLE_REF_FRAME_MVS, 0);
  CTRL(AV1E_SET_ENABLE_WARPED_MOTION, 0);
  CTRL(AV1E_SET_ENABLE_OBMC, 0);
  CTRL(AV1E_SET_ENABLE_GLOBAL_MOTION, 0);
  CTRL(AOME_SET_ENABLEAUTOALTREF, 0);
  CTRL(AV1E_SET_ENABLE_TPL_MODEL, 0);
  CTRL(AV1E_SET_AQ_MODE, 0);
  aom_image_t *img=aom_img_alloc(NULL, AOM_IMG_FMT_I420, w, h, 1);
  if (!img) fail("image allocation failed");
  fwrite("DKIF\0\0\40\0AV01", 1, 12, stdout);
  putchar(w&255); putchar(w>>8); putchar(h&255); putchar(h>>8);
  write32(30); write32(1); write32(count); write32(0);
  for (unsigned f=0; f<count; ++f) {
    codec_compute_mode=read32();
    unsigned target=read32(), ref0=read32(), ref1=read32();
    codec_compute_dx=(int32_t)read32(); codec_compute_dy=(int32_t)read32();
    if (codec_compute_mode>2 || target>7 || ref0>7 || ref1>7) fail("invalid instruction");
    for(unsigned y=0; y<h; ++y) {
      unsigned char *row=img->planes[0]+y*img->stride[0];
      if (!codec_compute_mode) {
        if(fread(row,1,w,stdin)!=w) fail("short input plane");
      } else memset(row,0,w);
    }
    for(int p=1;p<3;++p) for(unsigned y=0;y<h/2;++y)
      memset(img->planes[p]+y*img->stride[p],128,w/2);
    aom_svc_ref_frame_config_t refs={0};
    for(int i=0;i<7;++i) refs.ref_idx[i]=ref0;
    refs.ref_idx[3]=target; /* Refresh target must appear in ref_idx. */
    refs.ref_idx[6]=ref1;
    refs.reference[0]=1;
    refs.reference[6]=codec_compute_mode==2;
    refs.refresh[target]=1;
    CTRL(AV1E_SET_SVC_REF_FRAME_CONFIG, &refs);
    check(&ctx,aom_codec_encode(&ctx,img,f,1,AOM_EFLAG_NO_UPD_ENTROPY | AOM_EFLAG_SET_PRIMARY_REF_NONE));
    aom_codec_iter_t iter=NULL;
    const aom_codec_cx_pkt_t *pkt;
    unsigned packets=0;
    while((pkt=aom_codec_get_cx_data(&ctx,&iter))) {
      if(pkt->kind!=AOM_CODEC_CX_FRAME_PKT) continue;
      write32(pkt->data.frame.sz); write32(f); write32(0);
      fwrite(pkt->data.frame.buf,1,pkt->data.frame.sz,stdout);
      ++packets;
    }
    if(packets!=1) fail("expected exactly one packet per frame");
  }
  aom_img_free(img); check(&ctx,aom_codec_destroy(&ctx));
  return ferror(stdout) ? 1 : 0;
}
