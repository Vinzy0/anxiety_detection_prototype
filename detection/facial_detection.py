import numpy as np
import tensorflow as tf

WINDOW_FRAMES = 30
N_FEATURES = 28

STD_INDICES = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 22, 26}

FACIAL_TENSION_THRESHOLD = 0.58

class FacialTensionDetector:
    def __init__(self, model_path, scaler_mean_path, scaler_std_path):
        self.buffer = []

        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        # Load global scaler
        self.scaler_mean = np.load(scaler_mean_path)
        self.scaler_std  = np.load(scaler_std_path)
        self.subject_mean = np.zeros(N_FEATURES, dtype=np.float32)
        self.subject_std  = np.ones(N_FEATURES, dtype=np.float32)

    def _stat(self, arr):
        a = np.asarray(arr, dtype=np.float32)
        return float(np.mean(a)), float(np.std(a))

    def _extract_features(self, rows):
        if not rows:
            return None

        def col(key, default=0.0):
            return [r.get(key, default) for r in rows]

        detection_rate = float(np.mean(col("face_detected", 1.0)))
        if detection_rate < 0.60:
            return None

        # Geometry
        bg_l = [r.get("lm159_y", 0) - r.get("lm55_y", 0) for r in rows]
        bg_r = [r.get("lm386_y", 0) - r.get("lm285_y", 0) for r in rows]
        bg_avg = [(a + b) / 2 for a, b in zip(bg_l, bg_r)]
        ibd = [abs(r.get("lm285_x", 0) - r.get("lm55_x", 0)) for r in rows]

        # Blendshapes
        bdl = col("bs_browDownLeft")
        bdr = col("bs_browDownRight")
        biu = col("bs_browInnerUp")
        esl = col("bs_eyeSquintLeft")
        esr = col("bs_eyeSquintRight")
        jaw = col("bs_jawOpen")
        mpl = col("bs_mouthPressLeft")
        mpr = col("bs_mouthPressRight")

        # Derived Signals
        bt = [(a + b) / 2 - 0.5 * c for a, b, c in zip(bdl, bdr, biu)]
        es_avg = [(a + b) / 2 for a, b in zip(esl, esr)]
        comp = [(a + b + c + d + (1 - e) + f) / 6
                for a, b, c, d, e, f in zip(esl, esr, mpl, mpr, jaw, bt)]

        # Stats
        bg_l_m, bg_l_s = self._stat(bg_l)
        bg_r_m, bg_r_s = self._stat(bg_r)
        bg_avg_m, bg_avg_s = self._stat(bg_avg)
        ibd_m, ibd_s = self._stat(ibd)

        bdl_m, bdl_s = self._stat(bdl)
        bdr_m, bdr_s = self._stat(bdr)
        biu_m, biu_s = self._stat(biu)
        bt_m, bt_s = self._stat(bt)

        esl_m, esl_s = self._stat(esl)
        esr_m, esr_s = self._stat(esr)
        es_avg_m, _ = self._stat(es_avg)

        jaw_m, jaw_s = self._stat(jaw)
        mpl_m, _ = self._stat(mpl)
        mpr_m, _ = self._stat(mpr)

        comp_m, comp_s = self._stat(comp)

        # Feature Vectors
        return [
            bg_l_m, bg_l_s,
            bg_r_m, bg_r_s,
            bg_avg_m, bg_avg_s,
            ibd_m, ibd_s,
            bdl_m, bdl_s,
            bdr_m, bdr_s,
            biu_m, biu_s,
            bt_m, bt_s,
            esl_m, esl_s,
            esr_m, esr_s,
            es_avg_m,
            jaw_m, jaw_s,
            mpl_m, mpr_m,
            comp_m, comp_s,
            detection_rate,
        ]

    def predict(self, row):

        self.buffer.append(row)
        if len(self.buffer) > WINDOW_FRAMES:
            self.buffer.pop(0)

        if len(self.buffer) < WINDOW_FRAMES:
            return None

        feats = self._extract_features(self.buffer)
        if feats is None:
            return None

        x = np.array(feats, dtype=np.float32)

        # Subject normalization
        for i in range(N_FEATURES):
            if i not in STD_INDICES:
                x[i] = (x[i] - self.subject_mean[i]) / self.subject_std[i]

        # Global scaler
        x = (x - self.scaler_mean) / self.scaler_std
        x = x.astype(np.float32)

        # TFLite
        self.interpreter.set_tensor(self.input_details['index'], x[np.newaxis, :])
        self.interpreter.invoke()
        prob = float(self.interpreter.get_tensor(self.output_details['index'])[0][0])

        return {
            "probability": prob,
            "tense": prob >= FACIAL_TENSION_THRESHOLD
        }