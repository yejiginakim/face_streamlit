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
    st.markdown("### 카테고리 선택 ")
    use_gender = st.multiselect('성별', ['female', 'male', 'unisex'], placeholder = '선택하세요')
    use_kind = st.multiselect('분류', ['fashion', 'sports'], default = ['fashion'], placeholder = '선택하세요')

# 예: 플래그로 사용
is_female = 'female' in use_gender
is_male   = 'male'   in use_gender
is_unisex = 'unisex' in use_gender
is_fashion = 'fashion' in use_kind
is_sports  = 'sports'  in use_kind

# 예: 세션에 저장(다른 페이지/콜백에서도 사용)
st.session_state['use_gender'] = use_gender
st.session_state['use_kind']   = use_kind



# (선택) 세션 키로도 보관
st.session_state['use_gender'] = use_gender
st.session_state['use_kind']   = use_kind

# 5) 실행 버튼: 두 그룹 모두 최소 1개 선택돼야 활성화
disabled = not (use_gender and use_kind)
run = st.button('실행', disabled=disabled)
if disabled:
    st.warning('성별과 분류에서 각각 최소 1개 이상 선택하세요.')
elif run:
    st.success(f'실행! 성별={use_gender}, 분류={use_kind}')
    # TODO: 실제 처리 로직if err_msgs:
    st.error("초기 임포트 경고가 있어요. 아래 로그를 확인하세요.")
    st.code("\n".join(err_msgs), language="text")
    
# ---------- 임포트 실패 시, 여기서 멈추지 말고 안내만 ----------
if err_msgs:
    st.info("위 임포트 문제를 해결해야 합성이 진행됩니다. (requirements.txt / OpenCV headless / vision.py 함수 확인)")
    # 업로드 위젯은 이미 보이므로, 여기서 바로 return 느낌으로 종료
    st.stop()

st.divider()



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
glob Antena_01.*={glob.glob('frames/images/SF191SKN_004_61 .*')}
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


# ---------- PD/자세/스케일/합성 ----------
# 1) PD_px / mid (그리고 눈선 기반 roll)
try:
    pd_px, eye_roll_deg, mid = vision.detect_pd_px(face_bgr)
except Exception as e:
    st.error(f"MediaPipe 계산 실패: {e}")
    st.stop()

if pd_px is None:
    st.error("얼굴/눈 검출 실패. 정면, 밝은 조명에서 다시 시도해 주세요.")
    st.stop()

# 2) (있으면) 3축 자세 가져오기 → 없으면 roll은 눈선 값으로
yaw = pitch = roll = None
if hasattr(vision, "head_pose_ypr"):
    try:
        yaw, pitch, roll = vision.head_pose_ypr(face_bgr)  # 각도 단위: °
    except Exception:
        yaw = pitch = roll = None
if roll is None:
    roll = eye_roll_deg

st.write(
    f"**PD_px**: {pd_px:.2f} px  /  "
    f"**roll**: {roll:.2f}°{' (eye-line)' if yaw is None else ''}  /  "
    f"**mid**: {tuple(round(v,1) for v in mid)}"
)

# 3) 프레임 PNG 클린업(흰 배경 제거 + 여백 트림)
fg_bgra = vision.remove_white_to_alpha(fg_bgra, thr=240)
fg_bgra = vision.trim_transparent(fg_bgra, pad=8)

# 4) px/mm 및 목표 총폭(px) 계산
mm_per_px = (PD_MM / pd_px) if PD_MM else None  # 1픽셀당 mm
if mm_per_px:
    st.write(f"**mm_per_px**: {mm_per_px:.4f}")
    target_total_px = (GCD / mm_per_px) * k     # 실제 mm를 픽셀로 변환
else:
    st.warning("PD(mm)가 없어 근사 스케일로 합성합니다. (TOTAL/GCD 비율 사용)")
    target_total_px = pd_px * k

# (옵션) yaw가 크면 살짝 가로 축소(원근 보정 느낌)
yaw_abs = abs(yaw) if yaw is not None else 0.0
yaw_scale = 1.0 - min(yaw_abs, 25.0) * 0.01   # 최대 25°에서 25% 축소
yaw_scale = max(0.75, yaw_scale)              # 과도 축소 방지

# 5) 리사이즈
h0, w0 = fg_bgra.shape[:2]
scale = (target_total_px / w0) * scale_mult * yaw_scale
new_size = (max(1, int(w0*scale)), max(1, int(h0*scale)))
fg_scaled = cv2.resize(fg_bgra, new_size, interpolation=cv2.INTER_LINEAR)

# 6) 회전(roll 사용)
M = cv2.getRotationMatrix2D((fg_scaled.shape[1]/2, fg_scaled.shape[0]/2), roll, 1.0)
fg_rot = cv2.warpAffine(
    fg_scaled, M, (fg_scaled.shape[1], fg_scaled.shape[0]),
    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
)

# (옵션) pitch가 아래(+)/위(-)면 세로 오프셋 조금 보정
pitch_deg = pitch if pitch is not None else 0.0
pitch_dy  = int(pitch_deg * 0.8)  # 0.5~1.2 사이 취향대로

# 7) 위치(브리지 중심을 mid에 정렬) + 미세조정
gx = int(mid[0] - fg_rot.shape[1] / 2) + dx
gy = int(mid[1] - fg_rot.shape[0] / 2) + dy + pitch_dy

# 8) 합성
out = vision.overlay_rgba(face_bgr.copy(), fg_rot, gx, gy)
show_image_bgr(out, caption="합성 결과")


# ---------- 다운로드 ----------
try:
    from io import BytesIO
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    st.download_button("결과 PNG 다운로드", data=buf.getvalue(),
                       file_name="SF191SKN_004_61.png", mime="image/png")
except Exception as e:
    st.warning(f"다운로드 준비 중 경고: {e}")
