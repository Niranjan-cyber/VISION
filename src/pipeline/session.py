"""
Reusable VISION pipeline session.

This module owns the exact same detection, tracking, face recognition, ANPR,
and event-intelligence orchestration that src/main.py's CLI loop used to run
inline. It is a faithful extraction (not a rewrite) of that per-frame logic
into a class both the CLI and the FastAPI backend can drive identically, so
there is exactly one implementation of the AI orchestration in the project.

No detection/tracking/face/ANPR/event *algorithms* live here — this module
only wires up the existing src/detection, src/tracking, src/face, src/anpr,
and src/events components in the same sequence main.py always used.
"""
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import cv2
import numpy as np

from src.core.types import (
    BoundingBox,
    Detection,
    FaceDetection,
    FaceEmbedding,
    IdentityMatch,
    PlateRecognitionResult,
    Track,
)
from src.anpr import (
    LicensePlateDetector,
    PlateEnhancer,
    PlateTrackCache,
    clean_plate_text,
    get_plate_ocr_engine,
    map_crop_to_global_bbox,
)
from src.detection.detector import YOLODetector
from src.events import (
    Alert,
    EventEngine,
    ObjectState,
    SecurityEvent,
    Zone,
    load_zones_from_file,
    point_in_zone,
)
from src.face.alignment import align_face
from src.face.association import FaceTrackAssociation, associate_faces_to_tracks
from src.face.detector import FaceDetector
from src.face.embedder import ONNXRuntimeArcFaceEmbedder, OpenCVArcFaceEmbedder
from src.face.gallery import load_gallery_from_dir
from src.face.matcher import FaceMatcher
from src.face.modern_embedder import W600KR50Embedder
from src.ingestion.video import VideoSource
from src.tracking.tracker import ByteTrackTracker

TARGET_VEHICLE_CLASSES: Set[str] = {"car", "truck", "bus", "motorcycle"}


class PipelineSubsystemError(RuntimeError):
    """Raised when a required pipeline subsystem fails to initialize, naming which one."""

    def __init__(self, subsystem: str, message: str):
        self.subsystem = subsystem
        super().__init__(message)


@dataclass
class FrameResult:
    """Single-frame output of PipelineSession.process_frame() — the normalized
    handoff point both the CLI's drawing code and the backend's serializer consume."""

    frame_index: int
    timestamp: float
    detections: List[Detection]
    tracks: List[Track]
    faces: List[FaceDetection]
    associations: List[FaceTrackAssociation]
    plates: List[PlateRecognitionResult]
    object_states: List[ObjectState]
    new_events: List[SecurityEvent]
    new_alerts: List[Alert]
    breached_zone_ids: Set[str]
    ran_inference: bool


class PipelineSession:
    """
    Owns one video source plus every AI subsystem (detector, tracker, face
    recognition stack, optional ANPR stack, optional event engine) and the
    cumulative per-track state that must persist across frames (identity
    cache, plate cache, event history).
    """

    def __init__(
        self,
        video_path: str,
        model: str = "yolo11n.pt",
        confidence: float = 0.25,
        face_confidence: float = 0.50,
        face_threshold: float = 0.60,
        face_margin: float = 0.10,
        gallery_dir: str = "data/face_gallery",
        db_uri: Optional[str] = None,
        interval: int = 1,
        face_model: str = "w600k_r50",
        arcface_backend: str = "onnxruntime",
        enable_anpr: bool = True,
        plate_model: Optional[str] = None,
        ocr_engine: str = "auto",
        zones_path: Optional[str] = None,
        loitering_duration: float = 30.0,
        stationary_duration: float = 60.0,
        movement_threshold: float = 15.0,
        camera_id: str = "BOP-01",
        debug_face_matching: bool = False,
        debug_face_alignment: bool = False,
        debug_face_crops: bool = False,
        verbose: bool = True,
    ):
        self.video_path = video_path
        self.model_name = model
        self.confidence = confidence
        self.face_confidence = face_confidence
        self.face_threshold = face_threshold
        self.face_margin = face_margin
        self.gallery_dir = gallery_dir
        self.db_uri = db_uri
        self.interval = interval
        self.face_model = face_model
        self.arcface_backend = arcface_backend
        self.plate_model = plate_model
        self.ocr_engine_name = ocr_engine
        self.zones_path = zones_path
        self.loitering_duration = loitering_duration
        self.stationary_duration = stationary_duration
        self.movement_threshold = movement_threshold
        self.camera_id = camera_id
        self.debug_face_matching = debug_face_matching
        self.debug_face_alignment = debug_face_alignment
        self.debug_face_crops = debug_face_crops
        self._aligned_debug_saved_count = 0
        self.verbose = verbose

        self.model_display_name = (
            "InsightFace W600K-R50"
            if face_model == "w600k_r50"
            else f"ArcFace R100 ({arcface_backend})"
        )

        # 1. Ingestion
        try:
            self.source = VideoSource(video_path)
        except (FileNotFoundError, ValueError) as e:
            raise PipelineSubsystemError("video", str(e)) from e
        self._log(
            f"[INFO] Video loaded successfully. Resolution: {self.source.width}x{self.source.height}, "
            f"FPS: {self.source.fps:.2f}, Total Frames: {self.source.frame_count}"
        )

        # 2. YOLODetector
        try:
            self.detector = YOLODetector(model_name=model, confidence_threshold=confidence)
        except Exception as e:
            self.source.release()
            raise PipelineSubsystemError("detection", str(e)) from e

        # 3. ByteTrackTracker
        try:
            self.tracker = ByteTrackTracker(track_thresh=confidence, match_thresh=0.8, track_buffer=30)
        except Exception as e:
            self.source.release()
            raise PipelineSubsystemError("tracking", str(e)) from e

        # 4. FaceDetector
        try:
            self.face_detector = FaceDetector(score_threshold=face_confidence)
        except Exception as e:
            self.source.release()
            raise PipelineSubsystemError("face_id", str(e)) from e

        # 5. FaceEmbedder
        try:
            if face_model == "w600k_r50":
                self.face_embedder = W600KR50Embedder(model_path="models/w600k_r50.onnx")
            elif arcface_backend == "opencv":
                self.face_embedder = OpenCVArcFaceEmbedder()
            else:
                self.face_embedder = ONNXRuntimeArcFaceEmbedder()
        except Exception as e:
            self.source.release()
            raise PipelineSubsystemError("face_id", str(e)) from e

        # 6. FaceGallery & FaceMatcher
        try:
            if db_uri:
                self._log(f"[INFO] Connecting to database: {db_uri}")
            else:
                self._log("[INFO] Persistence: in-memory (no db_uri supplied)")
            self.gallery = load_gallery_from_dir(gallery_dir, self.face_detector, self.face_embedder, db_uri=db_uri)
            self.face_matcher = FaceMatcher(self.gallery, threshold=face_threshold, margin=face_margin)
        except Exception as e:
            self._log(f"[WARNING] Gallery database initialization failed ({e}). Falling back to in-memory mode.")
            try:
                self.gallery = load_gallery_from_dir(gallery_dir, self.face_detector, self.face_embedder, db_uri=None)
                self.face_matcher = FaceMatcher(self.gallery, threshold=face_threshold, margin=face_margin)
            except Exception as fallback_err:
                self.source.release()
                raise PipelineSubsystemError("face_id", str(fallback_err)) from fallback_err

        # 7. ANPR (optional)
        self.enable_anpr = enable_anpr
        if enable_anpr:
            try:
                self.plate_detector = LicensePlateDetector(model_path=plate_model)
                self.plate_enhancer = PlateEnhancer(target_height=70)
                self.plate_ocr = get_plate_ocr_engine(ocr_engine)
                self.plate_track_cache = PlateTrackCache()
            except Exception as e:
                self._log(f"[WARNING] ANPR initialization failed ({e}). Disabling ANPR for this session.")
                self.enable_anpr = False
                self.plate_detector = None
                self.plate_enhancer = None
                self.plate_ocr = None
                self.plate_track_cache = None
        else:
            self.plate_detector = None
            self.plate_enhancer = None
            self.plate_ocr = None
            self.plate_track_cache = None

        # 8. Zones + EventEngine (optional)
        self.zones: List[Zone] = []
        self.event_engine: Optional[EventEngine] = None
        if zones_path:
            try:
                self.zones = load_zones_from_file(zones_path)
                self._log(f"[INFO] Loaded {len(self.zones)} surveillance zone(s) from '{zones_path}'.")
                self.event_engine = EventEngine(
                    zones=self.zones,
                    loitering_duration=loitering_duration,
                    stationary_duration=stationary_duration,
                    movement_threshold=movement_threshold,
                )
            except Exception as e:
                self._log(f"[WARNING] Failed to load zones from '{zones_path}': {e}. Event engine disabled.")

        # Cumulative per-track / per-session state
        self.track_identity_cache: Dict[int, IdentityMatch] = {}
        self.track_plate_map: Dict[int, PlateRecognitionResult] = {}
        self.latest_detections: List[Detection] = []
        self.latest_tracks: List[Track] = []
        self.latest_faces: List[FaceDetection] = []
        self.latest_associations: List[FaceTrackAssociation] = []
        self.latest_plates: List[PlateRecognitionResult] = []
        self.latest_object_states: List[ObjectState] = []
        self.latest_new_events: List[SecurityEvent] = []
        self.latest_new_alerts: List[Alert] = []
        self.latest_breached_zone_ids: Set[str] = set()

        self.frame_index = 0
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.recent_inference_fps = 0.0
        self.total_detections = 0
        self.total_faces_detected = 0
        self.total_faces_associated = 0
        self.total_embeddings_generated = 0
        self.total_recognized_faces = 0
        self.total_unknown_faces = 0
        self.total_plates_detected = 0
        self.observed_unique_track_ids: Set[int] = set()
        self.max_active_tracks = 0
        self.start_time = time.time()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)

    @property
    def status(self) -> Dict[str, bool]:
        """Real subsystem online/offline flags, derived from actual init state — never hardcoded."""
        return {
            "video": self.source is not None,
            "detection": self.detector is not None,
            "tracking": self.tracker is not None,
            "face_id": self.face_detector is not None and self.face_embedder is not None,
            "anpr": self.enable_anpr,
            "events": self.event_engine is not None,
        }

    def process_frame(self, frame: np.ndarray, frame_index: int) -> Optional[FrameResult]:
        """
        Runs one frame through detection -> tracking -> face ID -> ANPR ->
        event evaluation. Faithful port of src/main.py's original per-frame
        loop body — same frame-sampling gate, same caching behavior.
        """
        if frame is None:
            return None

        frame_h, frame_w = frame.shape[:2]
        ran_inference = (frame_index - 1) % self.interval == 0

        if ran_inference:
            t0 = time.time()

            # A. YOLO Detection
            self.latest_detections = self.detector.detect(frame)

            # B. ByteTrack Tracking
            self.latest_tracks = self.tracker.update(self.latest_detections, frame_index)

            # C. Person-Only Face Detection & Coordinate Conversion
            person_tracks = [t for t in self.latest_tracks if t.class_name == "person"]
            current_frame_faces: List[FaceDetection] = []

            for person in person_tracks:
                pb = person.bbox
                px1 = max(0, min(pb.x1, frame_w))
                py1 = max(0, min(pb.y1, frame_h))
                px2 = max(0, min(pb.x2, frame_w))
                py2 = max(0, min(pb.y2, frame_h))
                if px2 <= px1 or py2 <= py1:
                    continue

                person_crop = frame[py1:py2, px1:px2]
                crop_faces = self.face_detector.detect(person_crop)

                for crop_face in crop_faces:
                    fb = crop_face.bbox
                    gx1 = max(0, min(px1 + fb.x1, frame_w))
                    gy1 = max(0, min(py1 + fb.y1, frame_h))
                    gx2 = max(0, min(px1 + fb.x2, frame_w))
                    gy2 = max(0, min(py1 + fb.y2, frame_h))
                    if gx2 <= gx1 or gy2 <= gy1:
                        continue

                    global_landmarks = None
                    if crop_face.landmarks is not None:
                        global_landmarks = crop_face.landmarks.copy()
                        global_landmarks[:, 0] += px1
                        global_landmarks[:, 1] += py1

                    global_face = FaceDetection(
                        bbox=BoundingBox(x1=gx1, y1=gy1, x2=gx2, y2=gy2),
                        confidence=crop_face.confidence,
                        landmarks=global_landmarks,
                    )
                    current_frame_faces.append(global_face)

            self.latest_faces = current_frame_faces

            # D. Face-to-Track Association
            self.latest_associations = associate_faces_to_tracks(self.latest_tracks, self.latest_faces)

            # E. Face Recognition for Associated Faces
            frame_recognized_count = 0
            frame_unknown_count = 0

            for assoc in self.latest_associations:
                track_id = assoc.track_id

                if track_id in self.track_identity_cache:
                    match_result = self.track_identity_cache[track_id]
                else:
                    aligned_crop = None
                    if assoc.face.landmarks is not None:
                        aligned_crop = align_face(frame, assoc.face.landmarks)

                    if aligned_crop is not None:
                        face_input = aligned_crop
                    else:
                        fb = assoc.face.bbox
                        fx1 = max(0, min(fb.x1, frame_w))
                        fy1 = max(0, min(fb.y1, frame_h))
                        fx2 = max(0, min(fb.x2, frame_w))
                        fy2 = max(0, min(fb.y2, frame_h))
                        if fx2 <= fx1 or fy2 <= fy1:
                            continue
                        face_input = frame[fy1:fy2, fx1:fx2]

                    if face_input.size == 0:
                        continue

                    if self.debug_face_alignment and self._aligned_debug_saved_count < 5:
                        os.makedirs("data/debug/aligned_faces", exist_ok=True)
                        aligned_save_path = f"data/debug/aligned_faces/track_{track_id}_frame_{frame_index}.jpg"
                        cv2.imwrite(aligned_save_path, face_input)
                        self._aligned_debug_saved_count += 1
                        self._log(f"[DEBUG] Saved aligned face ({face_input.shape[1]}x{face_input.shape[0]}) to '{aligned_save_path}'")

                    if self.debug_face_crops:
                        os.makedirs("scratch/debug_face_crops", exist_ok=True)
                        crop_save_path = f"scratch/debug_face_crops/track_{track_id}_frame_{frame_index}.jpg"
                        cv2.imwrite(crop_save_path, face_input)

                    embedding = self.face_embedder.embed(face_input)
                    if embedding is not None:
                        self.total_embeddings_generated += 1
                        match_result = self.face_matcher.match(embedding)
                        self.track_identity_cache[track_id] = match_result

                        if self.debug_face_matching:
                            all_sims = self.face_matcher.get_all_similarities(embedding)
                            print(f"\n--- Diagnostic Matching for Track #{track_id} (Frame {frame_index}) ---", file=sys.stderr)
                            for id_name, id_score in sorted(all_sims.items(), key=lambda x: x[1], reverse=True):
                                print(f"  {id_name}: {id_score:.4f}", file=sys.stderr)
                            print(f"  Best       : {match_result.identity if match_result.is_match else 'None'} ({match_result.similarity:.4f})", file=sys.stderr)
                            print(f"  Margin     : {match_result.margin:.4f} (Required: >= {self.face_margin:.2f})", file=sys.stderr)

                        if not match_result.is_match:
                            self._log(f"[INFO] Track #{track_id}: face not recognized (no gallery match).")
                            if getattr(self.gallery, "db", None) is not None:
                                try:
                                    vec = embedding.vector if isinstance(embedding, FaceEmbedding) else embedding
                                    self.gallery.db.flag_unauthorized_user(
                                        embedding=vec,
                                        frame=face_input,
                                        track_id=track_id,
                                        video_source=self.video_path,
                                    )
                                except Exception as db_err:
                                    self._log(f"[WARNING] Failed to store flagged event in database: {db_err}")
                    else:
                        match_result = IdentityMatch(identity=None, similarity=0.0, is_match=False)

                if match_result.is_match:
                    frame_recognized_count += 1
                else:
                    frame_unknown_count += 1

            self.total_recognized_faces += frame_recognized_count
            self.total_unknown_faces += frame_unknown_count

            # F. Vehicle License Plate Recognition (ANPR)
            current_frame_plates: List[PlateRecognitionResult] = []

            if self.enable_anpr and self.plate_detector is not None:
                vehicle_tracks = [t for t in self.latest_tracks if t.class_name in TARGET_VEHICLE_CLASSES]
                for veh in vehicle_tracks:
                    tid = veh.track_id
                    cached_p = self.plate_track_cache.get(tid)

                    if cached_p is not None and cached_p.confidence >= 0.85:
                        current_frame_plates.append(cached_p)
                        self.track_plate_map[tid] = cached_p
                        continue

                    vb = veh.bbox
                    vx1 = max(0, min(vb.x1, frame_w))
                    vy1 = max(0, min(vb.y1, frame_h))
                    vx2 = max(0, min(vb.x2, frame_w))
                    vy2 = max(0, min(vb.y2, frame_h))

                    if (vx2 - vx1) < 30 or (vy2 - vy1) < 30:
                        if cached_p is not None:
                            current_frame_plates.append(cached_p)
                            self.track_plate_map[tid] = cached_p
                        continue

                    veh_crop = frame[vy1:vy2, vx1:vx2]
                    vh, vw = veh_crop.shape[:2]
                    plate_dets = self.plate_detector.detect(veh_crop)

                    best_candidate_res = None
                    for pdet in plate_dets[:3]:
                        pb = pdet.bbox
                        py1 = max(0, pb.y1 - 4)
                        py2 = min(vh, pb.y2 + 4)
                        px1 = max(0, pb.x1 - 6)
                        px2 = min(vw, pb.x2 + 6)
                        p_crop = veh_crop[py1:py2, px1:px2]
                        if p_crop.size == 0 or (px2 - px1) < 40 or (py2 - py1) < 12:
                            continue

                        enh_crop = self.plate_enhancer.enhance(p_crop)
                        target_crop = enh_crop if enh_crop is not None else p_crop
                        raw_txt, ocr_conf = self.plate_ocr.recognize(target_crop)
                        cln_txt, is_valid, mult = clean_plate_text(raw_txt)

                        if cln_txt and (is_valid or len(cln_txt) >= 7):
                            g_bbox = map_crop_to_global_bbox(
                                BoundingBox(x1=px1, y1=py1, x2=px2, y2=py2),
                                BoundingBox(x1=vx1, y1=vy1, x2=vx2, y2=vy2),
                                frame_w,
                                frame_h,
                            )
                            cand_res = PlateRecognitionResult(
                                raw_text=raw_txt,
                                cleaned_text=cln_txt,
                                confidence=ocr_conf * mult,
                                is_valid=is_valid,
                                bbox=g_bbox,
                            )
                            if is_valid and len(cln_txt) in {9, 10}:
                                best_candidate_res = cand_res
                                break
                            elif best_candidate_res is None or len(cln_txt) > len(best_candidate_res.cleaned_text):
                                best_candidate_res = cand_res

                    if best_candidate_res is not None:
                        updated_res = self.plate_track_cache.update(tid, best_candidate_res, frame_index)
                        current_frame_plates.append(updated_res)
                        self.track_plate_map[tid] = updated_res
                        self.total_plates_detected += 1
                    elif cached_p is not None:
                        current_frame_plates.append(cached_p)
                        self.track_plate_map[tid] = cached_p

                self.latest_plates = current_frame_plates

            t1 = time.time()
            inf_time = t1 - t0
            self.total_inference_time += inf_time
            self.inference_count += 1
            if inf_time > 0:
                self.recent_inference_fps = 1.0 / inf_time

            self.total_detections += len(self.latest_detections)
            self.total_faces_detected += len(self.latest_faces)
            self.total_faces_associated += len(self.latest_associations)
            for trk in self.latest_tracks:
                self.observed_unique_track_ids.add(trk.track_id)
            if len(self.latest_tracks) > self.max_active_tracks:
                self.max_active_tracks = len(self.latest_tracks)

        current_timestamp = (
            frame_index / self.source.fps if self.source.fps > 0 else frame_index * 0.033
        )

        # Unified ObjectState per active track — built every frame regardless of
        # the inference-sampling interval (matches original main.py behavior),
        # and used both for event evaluation and as the API/dashboard contract.
        object_states: List[ObjectState] = []
        for trk in self.latest_tracks:
            tid = trk.track_id
            match_info = self.track_identity_cache.get(tid)
            p_rec = self.track_plate_map.get(tid)

            has_face = match_info is not None
            if has_face:
                ident = match_info.identity if match_info.is_match else "UNKNOWN"
                sim = match_info.similarity
            else:
                ident = None
                sim = None

            plate_txt = p_rec.cleaned_text if p_rec is not None else None
            plate_conf = p_rec.confidence if p_rec is not None else None

            st = ObjectState(
                track_id=tid,
                object_type=trk.class_name,
                bbox=trk.bbox,
                confidence=trk.confidence,
                camera_id=self.camera_id,
                identity=ident,
                face_similarity=sim,
                has_face_detected=has_face,
                plate=plate_txt,
                plate_confidence=plate_conf,
                first_seen=current_timestamp,
                last_seen=current_timestamp,
            )
            object_states.append(st)

        breached_zone_ids: Set[str] = set()
        new_events: List[SecurityEvent] = []
        new_alerts: List[Alert] = []

        if self.event_engine is not None:
            new_events, new_alerts = self.event_engine.update(object_states, timestamp=current_timestamp)

            for alr in new_alerts:
                self._log(
                    f"\n[SECURITY ALERT] {alr.title} | Camera: {self.camera_id} | "
                    f"Track: #{alr.metadata.get('object_type', 'object')} | "
                    f"Zone: {alr.metadata.get('zone_name')} | Severity: {alr.severity.value}"
                )

            for z in self.zones:
                if z.zone_type == "restricted":
                    for obj in object_states:
                        if point_in_zone(obj.position, z):
                            breached_zone_ids.add(z.id)
                            break

        self.latest_object_states = object_states
        self.latest_new_events = new_events
        self.latest_new_alerts = new_alerts
        self.latest_breached_zone_ids = breached_zone_ids

        return FrameResult(
            frame_index=frame_index,
            timestamp=current_timestamp,
            detections=self.latest_detections,
            tracks=self.latest_tracks,
            faces=self.latest_faces,
            associations=self.latest_associations,
            plates=self.latest_plates,
            object_states=object_states,
            new_events=new_events,
            new_alerts=new_alerts,
            breached_zone_ids=breached_zone_ids,
            ran_inference=ran_inference,
        )

    def restart(self) -> None:
        """Reopens the video source from the beginning and resets per-run
        cumulative state, so the golden demo clip can loop continuously."""
        self.source.release()
        self.source = VideoSource(self.video_path)
        self.tracker.reset()
        self.track_identity_cache.clear()
        self.track_plate_map.clear()
        if self.enable_anpr:
            self.plate_track_cache = PlateTrackCache()
        if self.event_engine is not None:
            self.event_engine.reset()

        self.latest_detections = []
        self.latest_tracks = []
        self.latest_faces = []
        self.latest_associations = []
        self.latest_plates = []
        self.latest_object_states = []
        self.latest_new_events = []
        self.latest_new_alerts = []
        self.latest_breached_zone_ids = set()

        self.frame_index = 0
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.recent_inference_fps = 0.0
        self.total_detections = 0
        self.total_faces_detected = 0
        self.total_faces_associated = 0
        self.total_embeddings_generated = 0
        self.total_recognized_faces = 0
        self.total_unknown_faces = 0
        self.total_plates_detected = 0
        self.observed_unique_track_ids = set()
        self.max_active_tracks = 0
        self.start_time = time.time()

    def release(self) -> None:
        self.source.release()
