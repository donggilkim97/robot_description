| Depth model | Grasp model | Profile | Plane removal | Plane dist. | Above-plane th. | Plane min normal-z | DBSCAN eps | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Depth Anything V2 | pca | Depth Anything V2 / plane ON | True | 0.024 | 0.016 | 0.55 | 0.045 | Plane removal enabled because stronger table/background artefacts were observed |
| Depth Pro | pca | Depth Pro / plane OFF | True | 0.02 | 0.012 | 0.6 | 0.045 | Table-depth scaling + height filter + outlier removal + DBSCAN |
| ZoeDepth | pca | ZoeDepth / plane ON | True | 0.022 | 0.015 | 0.55 | 0.045 | Plane removal enabled because tilted planar artefacts were observed |
