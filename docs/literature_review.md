# Literature Review

## Scope

The assignment asks for algorithms and tools for low-altitude drone visual navigation in GNSS-denied conditions. The operational problem is:

> Given a reference flight with video and known positions, estimate the camera-center coordinate of a new query video without using query GNSS during inference.

The project combines ideas from:

- visual place recognition,
- local feature matching,
- homography verification,
- temporal sequence localization,
- optical-flow motion estimation,
- realtime tracking and reacquisition.

The final system uses a training-free Visual Place Recognition approach inspired by AnyLoc, then adds DJI-specific telemetry preprocessing and realtime region-anchor logic.

---

## 1. Visual Place Recognition

Visual Place Recognition (VPR) is the closest computer-vision formulation of the assignment. A database of reference images is built, each with a known position. A query image is then matched to the closest visual place.

This maps directly to the project:

| VPR concept        | Project equivalent                                 |
| ---                | ---                                                |
| Reference database | GNSS-tagged reference drone flights                |
| Query image        | New drone video frame                              |
| Retrieved place    | Estimated camera-center coordinate                 |
| Evaluation         | Compare estimate to query SRT-derived ground truth |

The main challenge is that low-altitude drone videos contain repeated or weakly distinctive structures such as roofs, roads, trees, fields, and buildings. These can look visually similar even when they correspond to different positions.

---

## 2. NetVLAD — Foundational Learning-Based VPR

**Paper:** Arandjelovic et al., "NetVLAD: CNN Features for Image Retrieval," CVPR 2016.  
**Code:** https://github.com/Relja/netvlad

NetVLAD introduced a differentiable VLAD aggregation layer for place recognition. It is a classic baseline for learned image retrieval and inspired many later supervised VPR methods.

However, NetVLAD depends on training data. For this project, training on the reference flights would be conceptually problematic because the reference area is also the evaluation area. The project therefore prefers a frozen, training-free descriptor.

---

## 3. MixVPR and CosPlace — Strong Supervised VPR

**MixVPR:** Ali-bey et al., "MixVPR: Feature Mixing for Visual Place Recognition," WACV 2023.  
**Code:** https://github.com/amaralibey/MixVPR

**CosPlace:** Berton et al., "Rethinking Visual Geo-Localization for Large-Scale Applications," CVPR 2022.  
**Code:** https://github.com/gmberton/CosPlace

MixVPR and CosPlace are strong supervised VPR approaches. They perform well on urban benchmarks, but supervised urban VPR does not necessarily transfer to low-altitude drone data.

The project does not train one of these models because:

1. The available project data is too small for robust supervised training.
2. Training on the reference flight risks overfitting.
3. The assignment is better framed as reference-map retrieval rather than model training.

---

## 4. AnyLoc — Main Methodological Inspiration

**Paper:** Keetha et al., "AnyLoc: Towards Universal Visual Place Recognition," RA-L / ICRA 2024.  
**Code:** https://github.com/AnyLoc/AnyLoc  
**PDF:** https://anyloc.github.io/assets/AnyLoc.pdf

AnyLoc is the central paper for this project. It proposes training-free visual place recognition using frozen foundation model features, especially DINOv2.

AnyLoc is relevant because:

- it does not require training on the target environment
- it works by building a reference image database
- it performs well across diverse environments
- it includes aerial/drone-style benchmarks

### What the project borrows from AnyLoc

| AnyLoc idea                   | Project implementation                                 |
| ---                           | ---                                                    |
| Training-free VPR             | No model finetuning on the drone videos                |
| Frozen DINOv2 features        | DINOv2 ViT-S/14 descriptors                            |
| Reference database            | Preprocessed reference flight frames                   |
| Query-to-reference retrieval  | DINO similarity search                                 |
| Domain-specific reference map | DJI reference flights with projected camera-center GPS |

### What differs from full AnyLoc

Full AnyLoc recommends VLAD aggregation over DINOv2 patch tokens. This project mostly uses mean-pooled DINOv2 descriptors because they were simpler, faster, and worked sufficiently well when combined with LightGlue and temporal/geometric filters.

VLAD with an aerial-domain vocabulary remains a future improvement.

---

## 5. DINOv2

**Paper:** Oquab et al., "DINOv2: Learning Robust Visual Features Without Supervision," 2023.  
**Code:** https://github.com/facebookresearch/dinov2

DINOv2 is the frozen visual backbone. It produces patch tokens from a Vision Transformer trained with self-supervised learning.

Why DINOv2 fits the project:

- no training required,
- strong semantic and spatial features,
- robust enough for retrieval across drone frames,
- compatible with AnyLoc-style VPR.

In the implementation, `src/anyloc_dino_retrieval.py` and related scripts extract DINOv2 patch descriptors and mean-pool them into global image descriptors.

---

## 6. SuperPoint and LightGlue

**SuperPoint paper:** DeTone et al., "SuperPoint: Self-Supervised Interest Point Detection and Description," CVPRW 2018.  
**LightGlue paper:** Lindenberger et al., "LightGlue: Local Feature Matching at Light Speed," ICCV 2023.  
**Code:** https://github.com/cvg/LightGlue

DINOv2 global retrieval finds visually similar frames, but global similarity alone can select wrong places. LightGlue checks local geometric evidence by matching SuperPoint keypoints between a query and a candidate reference image.

The project uses LightGlue to compute:

- number of matches,
- RANSAC inliers,
- inlier ratio,
- homography reprojection quality,
- projected center consistency,
- inlier spread diagnostics.

This local verification step is essential because many campus/drone scenes repeat visually.

---

## 7. Homography and RANSAC Verification

A homography maps points from one image plane to another when the scene is approximately planar or when the camera undergoes rotation around a center. Drone views are not perfectly planar, but homography still provides useful evidence for whether two images share a geometrically consistent region.

The project uses OpenCV RANSAC homography to separate LightGlue matches into inliers and outliers.

The final V7 update adds a spread-consistency check:

- If query and reference altitudes are similar, the distribution of inliers should have similar spatial spread in both images.
- If one image has widely spread inliers and the other has a tiny cluster, the match may be an object-level false positive rather than a good pose match.
- The check is altitude-scaled so that different flight heights are treated more fairly.

This directly addresses cases where two visually similar objects appear at different image scales because the query and reference flights were captured at different heights.

---
## Optional Scale-Aware Reference Preprocessing

In addition to altitude-aware inlier spread consistency, the project also includes an optional scale-aware reference preprocessing tool: `tools/make_scale_aware_reference_manifest.py`, which was used earlier.

This tool physically changes the reference images before retrieval. For each reference frame, it estimates a crop ratio from the target/query altitude and the reference altitude:

crop_ratio = target_altitude / reference_altitude

The script center-crops the reference image by this ratio and resizes the crop back to the original resolution. This simulates the apparent scale change caused by flying at a different height: when the target altitude is lower than the reference altitude, the reference image is zoomed in so objects appear closer to the query scale.

This is different from the latest spread-consistency check. Spread consistency does not alter the images; it only checks whether the LightGlue/RANSAC inlier geometry is consistent with the expected altitude-induced scale difference. The scale-aware manifest tool is a stronger preprocessing step because it changes what DINOv2 and LightGlue see.

In the current final pipeline, scale-aware reference preprocessing is not enabled by default. It should be treated as an optional experiment and compared against the normal run before being promoted to the final pipeline.

---
## 8. Temporal Sequence Localization

A drone flight is a sequence, not a set of independent images. Consecutive estimates should not jump randomly across the map.

The offline pipeline used Motion Viterbi:

```text
DINOv2 top-k candidates
→ LightGlue candidate scores
→ Viterbi path selection with motion penalties
```

This is conceptually related to SeqSLAM, which showed that sequence consistency can greatly improve localization even when individual image matches are noisy.

The offline pipeline achieved the best numerical accuracy because Viterbi can optimize over the whole sequence. But Viterbi is not fully realtime because it requires future frames.

---

## 9. Realtime Region Anchors

For realtime inference, the system cannot wait for the whole flight. The final pipeline therefore uses region anchors:

1. **Acquire:** search globally until a region gets enough visual/geometric support.
2. **Lock:** search locally around the accepted region.
3. **Recover/Reacquire:** if confidence falls, output `NO_ESTIMATE` and widen/globalize search.

This converts offline sequence localization into a causal tracking system.

The key design decision is that the system is allowed to abstain. It is better to output `NO_ESTIMATE` than to publish a likely wrong coordinate.

---

## 10. Optical Flow

Optical flow estimates image motion between consecutive frames. The project tested flow as a dead-reckoning cue, but pure flow is not enough for global localization because:

- pixel flow depends on altitude,
- camera rotation and gimbal motion distort motion,
- parallax changes with scene depth,
- flow does not directly provide a global map direction.

The final pipeline uses flow only as a bounded short-gap cue and can output `NO_ESTIMATE` when confidence is low. It does not trust flow as the final GPS estimate.

---

## 11. Why We Did Not Train a Model

Training on the reference flights would make the evaluation less clean: the same area would be used for training and testing. The assignment is better solved by building a reference map and retrieving against it.

This is exactly the advantage of AnyLoc-style frozen features: the reference flight acts as a database, not as a supervised training set.

---

## 12. How the Literature Maps to the Final System

| Literature idea                 | Final system component                          |
| ---                             | ---                                             |
| AnyLoc / training-free VPR      | Frozen DINOv2 reference/query descriptors       |
| DINOv2                          | Global retrieval feature backbone               |
| SuperPoint + LightGlue          | Local feature verification                      |
| Homography + RANSAC             | Geometric consistency filter                    |
| SeqSLAM / temporal consistency  | Offline Viterbi and realtime region persistence |
| Optical flow                    | Short-gap motion prior only                     |
| Realtime tracking/reacquisition | Region-anchor V7 state machine                  |
| Safety/abstention               | `NO_ESTIMATE` frames                            |

---

## 13. Limitations and Future Work

### AnyLoc VLAD

The most direct improvement is to implement the stronger AnyLoc configuration: DINOv2 patch tokens aggregated with VLAD and an aerial-domain vocabulary.

### Better camera pose metadata

Exact yaw/gimbal metadata would improve both ground-truth projection and query/reference pose reasoning.

### Landmark-aware localization

A visually correct match can still be pose-wrong. Future work could explicitly estimate landmark position and camera pose rather than copying reference-frame pose.

### Faster realtime deployment

LightGlue remains the compute bottleneck. A deployed system could run DINO every frame but LightGlue only on selected candidate frames.

---

## Conclusion

The literature supports the final design: frozen DINOv2 features provide a training-free VPR foundation, LightGlue and homography improve local reliability, and temporal/region logic turns frame retrieval into a realtime navigation system. The final V7 spread-consistency pipeline is therefore a practical adaptation of modern VPR and local feature matching to GNSS-denied drone navigation.
