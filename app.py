# ---------- 반드시 최상단 1회 ----------
import streamlit as st
st.set_page_config(page_title="iPhone PD → 선글라스 합성 (Antena_01)", layout="wide")

# ---------- 기본 진단 캡션 ----------
import sys, platform, os, glob
st.caption(f"Python: {sys.version.split()[0]} / Arch: {platform.machine()} / CWD: {os.getcwd()}")

# ---------- 유틸: 이미지 표시(버전 호환) ----------
def show_image_bgr(img_bgr, **kwargs):
    try:
        import cv2
        import numpy as np
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            st.image(rgb, use_container_width=True, **kwargs)
        except TypeError:
            try:
                st.image(rgb, use_column_width=True, **kwargs)
            except TypeError:
                st.image(rgb, **kwargs)
    except Exception as e:
        st.error(f"이미지 표시 중 오류: {e}")

# ---------- 지연 임포트: 실패해도 UI는 뜨게 ----------
cv2 = np = Image = None
vision = None
err_msgs = []

try:
    import numpy as np
except Exception as e:
    err_msgs.append(f"numpy import 실패: {e}")

try:
    import cv2
except Exception as e:
    err_msgs.append(f"opencv(cv2) import 실패: {e}")

try:
    from PIL import Image
except Exception as e:
    err_msgs.append(f"Pillow import 실패: {e}")

try:
    import vision  # vision.py 에 detect_pd_px / load_fixed_antena / overlay_rgba 있어야 함
except Exception as e:
    err_msgs.append(f"vision 임포트 실패: {e}")

# ---------- 사이드바 / 입력 UI는 무조건 출력 ----------
st.title("🧍→🕶️ Antena_01 합성 (GCD 앵커) — 안전모드")

with st.sidebar:
    st.subheader("PD (mm)")
    # URL ?pd_mm=... 혹은 ?pd=...
    params = st.query_params
    def fget(k, default=None):
        try:
            v = params.get(k, None)
            return float(v) if v not in (None, "") else default
        except Exception:
            return default
    pd_from_url = fget("pd_mm", fget("pd", None))
    PD_MM = st.number_input("PD (mm) 직접 입력", value=pd_from_url or 63.0, step=0.1, format="%.3f")

    st.subheader("미세 조정")
    dx = st.slider("수평 오프셋(px)", -200, 200, 0)
    dy = st.slider("수직 오프셋(px)", -200, 200, 0)
    scale_mult = st.slider("스케일 보정(배)", 0.8, 1.2, 1.0)

colL, colR = st.columns(2)
with colL:
    st.markdown("### 1) 얼굴 사진 업로드")
    img_file = st.file_uploader("정면 얼굴 사진", type=["jpg","jpeg","png"])

with colR:
    st.markdown("### 2) 결과/수치")
    if err_msgs:
        st.error("초기 임포트 경고가 있어요. 아래 로그를 확인하세요.")
        st.code("\n".join(err_msgs), language="text")

# ---------- 임포트 실패 시, 여기서 멈추지 말고 안내만 ----------
if err_msgs:
    st.info("위 임포트 문제를 해결해야 합성이 진행됩니다. (requirements.txt / OpenCV headless / vision.py 함수 확인)")
    # 업로드 위젯은 이미 보이므로, 여기서 바로 return 느낌으로 종료
    st.stop()

# ---------- 프레임 로드 ----------
try:
    fg_bgra, dims = vision.load_fixed_antena()
except Exception as e:
    st.error(f"프레임 로드 호출 실패: {e}")
    dims = None
    fg_bgra = None

if fg_bgra is None or dims is None:
    st.error("프레임 이미지를 읽을 수 없어요. 경로/포맷을 확인해 주세요.")
    st.code(f"""
exists(frames)={os.path.isdir('frames')}
exists(frames/images)={os.path.isdir('frames/images')}
list(frames/images)[:10]={os.listdir('frames/images')[:10] if os.path.isdir('frames/images') else 'N/A'}
glob Antena_01.*={glob.glob('frames/images/Antena_01.*')}
    """, language="text")
    st.stop()

A, DBL, TOTAL = dims
GCD = A + DBL
k = (TOTAL / GCD) if GCD else 2.0
st.caption(f"프레임 치수 A={A}, DBL={DBL}, TOTAL={TOTAL} (GCD={GCD}, k=TOTAL/GCD={k:.3f})")

# ---------- 얼굴 이미지 업로드 필요 ----------
if not img_file:
    st.info("얼굴 사진을 업로드하면 합성을 시작합니다.")
    st.stop()

# ---------- 얼굴 이미지 읽기 ----------
try:
    file_bytes = np.frombuffer(img_file.read(), dtype=np.uint8)
    face_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if face_bgr is None:
        raise RuntimeError("OpenCV가 이미지를 디코드하지 못함")
except Exception as e:
    st.error(f"얼굴 이미지 로드 실패: {e}")
    st.stop()

# ---------- PD_px / 각도 / 중점 ----------
try:
    pd_px, angle_deg, mid = vision.detect_pd_px(face_bgr)
except Exception as e:
    st.error(f"MediaPipe 계산 실패: {e}")
    st.stop()

if pd_px is None:
    st.error("얼굴/눈 검출 실패. 정면, 밝은 조명에서 다시 시도해 주세요.")
    st.stop()

st.write(f"**PD_px**: {pd_px:.2f} px  /  **angle**: {angle_deg:.2f}°  /  **mid**: {tuple(round(v,1) for v in mid)}")

# ---------- 스케일 계산 ----------
px_per_mm = (pd_px / PD_MM) if PD_MM else None
if px_per_mm:
    st.write(f"**px_per_mm**: {px_per_mm:.4f}")
    target_total_px = (GCD * px_per_mm) * k
else:
    st.warning("PD(mm)가 없어 근사 스케일로 합성합니다. (TOTAL/GCD 비율 사용)")
    target_total_px = pd_px * k

# ---------- 리사이즈/회전/합성 ----------
try:
    h0, w0 = fg_bgra.shape[:2]
    scale = (target_total_px / w0) * scale_mult
    new_size = (max(1, int(w0*scale)), max(1, int(h0*scale)))
    fg_scaled = cv2.resize(fg_bgra, new_size, interpolation=cv2.INTER_LINEAR)

    M = cv2.getRotationMatrix2D((fg_scaled.shape[1]/2, fg_scaled.shape[0]/2), angle_deg, 1.0)
    fg_rot = cv2.warpAffine(
        fg_scaled, M, (fg_scaled.shape[1], fg_scaled.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
    )

    gx = int(mid[0] - fg_rot.shape[1] / 2) + dx
    gy = int(mid[1] - fg_rot.shape[0] / 2) + dy
    out = vision.overlay_rgba(face_bgr.copy(), fg_rot, gx, gy)

    show_image_bgr(out, caption="합성 결과")
except Exception as e:
    st.error(f"합성 중 오류: {e}")
    st.stop()

# ---------- 다운로드 ----------
try:
    from io import BytesIO
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    st.download_button("결과 PNG 다운로드", data=buf.getvalue(),
                       file_name="Antena_01_result.png", mime="image/png")
except Exception as e:
    st.warning(f"다운로드 준비 중 경고: {e}")
