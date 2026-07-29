# Generated large-data result summary

This file is generated from the fetched JSON logs. It reports exact best-observed values together with a predeclared 5% relative comparability band; a small numerical difference is not converted into a categorical method lead.
The reported Betti-DTW values use the suite's recorded FastDTW approximation; H0 bottleneck distances use the exact sorted zero-birth-bar specialization and H1 uses exact compiled GUDHI.

## 10x mouse brain

### All evaluated methods and references

- fastest steady-state runtime: PCA (2D) (0.148147)
- highest local kNN overlap: cuML t-SNE (0.347653)
- highest centroid-distance correlation: PCA (2D) (0.68317)
- highest centroid-adjacency recall: cuML UMAP (0.483333)
- highest balanced context accuracy: cuML UMAP (0.64303)
- lowest diameter-normalized beta-0 bottleneck distance: non-discriminating equality at 0.427075 across PCA (2D), DiRe (auto cuVS), DiRe (IVF-Flat control), DiRe (spectral init), DiRe (topology preset), cuML UMAP, cuML t-SNE, Cell Ranger t-SNE
- lowest diameter-normalized beta-1 bottleneck distance: cuML t-SNE (0.01414)
- lowest beta-0 DTW: cuML t-SNE (0.1297)
- lowest beta-1 DTW: best observed value 0.038475; practically comparable within 5%: Cell Ranger t-SNE, cuML t-SNE

### Fresh production-policy nonlinear methods on the same GPU

- fastest steady-state runtime: DiRe (auto cuVS) (7.81473)
- highest local kNN overlap: cuML t-SNE (0.347653)
- highest centroid-distance correlation: cuML t-SNE (0.200124)
- highest centroid-adjacency recall: cuML UMAP (0.483333)
- highest balanced context accuracy: cuML UMAP (0.64303)
- lowest diameter-normalized beta-0 bottleneck distance: non-discriminating equality at 0.427075 across DiRe (auto cuVS), cuML UMAP, cuML t-SNE
- lowest diameter-normalized beta-1 bottleneck distance: cuML t-SNE (0.01414)
- lowest beta-0 DTW: cuML t-SNE (0.1297)
- lowest beta-1 DTW: cuML t-SNE (0.040075)

### Same-GPU nonlinear methods including forced-index and predeclared DiRe sensitivity controls

- fastest steady-state runtime: DiRe (auto cuVS) (7.81473)
- highest local kNN overlap: cuML t-SNE (0.347653)
- highest centroid-distance correlation: cuML t-SNE (0.200124)
- highest centroid-adjacency recall: cuML UMAP (0.483333)
- highest balanced context accuracy: cuML UMAP (0.64303)
- lowest diameter-normalized beta-0 bottleneck distance: non-discriminating equality at 0.427075 across DiRe (auto cuVS), DiRe (IVF-Flat control), DiRe (spectral init), DiRe (topology preset), cuML UMAP, cuML t-SNE
- lowest diameter-normalized beta-1 bottleneck distance: cuML t-SNE (0.01414)
- lowest beta-0 DTW: cuML t-SNE (0.1297)
- lowest beta-1 DTW: cuML t-SNE (0.040075)

## arXiv corpus

### All evaluated methods and references

- fastest steady-state runtime: PCA (2D) (0.980988)
- highest local kNN overlap: best observed value 0.0991467; practically comparable within 5%: DiRe (topology preset), cuML UMAP
- highest centroid-distance correlation: best observed value 0.873422; practically comparable within 5%: PCA (2D), cuML UMAP
- highest centroid-adjacency recall: cuML UMAP (0.75)
- highest balanced context accuracy: cuML UMAP (0.320419)
- lowest diameter-normalized beta-0 bottleneck distance: best observed value 0.35112; practically comparable within 5%: cuML t-SNE, DiRe (spectral init), PCA (2D), DiRe (auto cuVS), DiRe (IVF-Flat control), DiRe (topology preset), cuML UMAP
- lowest diameter-normalized beta-1 bottleneck distance: best observed value 0.0245941; practically comparable within 5%: PCA (2D), DiRe (topology preset), cuML UMAP, cuML t-SNE, DiRe (spectral init), DiRe (IVF-Flat control)
- lowest beta-0 DTW: PCA (2D) (3.17135)
- lowest beta-1 DTW: DiRe (topology preset) (0.11005)

### Fresh production-policy nonlinear methods on the same GPU

- fastest steady-state runtime: DiRe (auto cuVS) (13.6931)
- highest local kNN overlap: cuML UMAP (0.09908)
- highest centroid-distance correlation: best observed value 0.845941; practically comparable within 5%: cuML UMAP, DiRe (auto cuVS)
- highest centroid-adjacency recall: cuML UMAP (0.75)
- highest balanced context accuracy: cuML UMAP (0.320419)
- lowest diameter-normalized beta-0 bottleneck distance: best observed value 0.35112; practically comparable within 5%: cuML t-SNE, DiRe (auto cuVS), cuML UMAP
- lowest diameter-normalized beta-1 bottleneck distance: best observed value 0.0245941; practically comparable within 5%: cuML UMAP, cuML t-SNE
- lowest beta-0 DTW: DiRe (auto cuVS) (3.46995)
- lowest beta-1 DTW: DiRe (auto cuVS) (0.162375)

### Same-GPU nonlinear methods including forced-index and predeclared DiRe sensitivity controls

- fastest steady-state runtime: DiRe (auto cuVS) (13.6931)
- highest local kNN overlap: best observed value 0.0991467; practically comparable within 5%: DiRe (topology preset), cuML UMAP
- highest centroid-distance correlation: best observed value 0.845941; practically comparable within 5%: cuML UMAP, DiRe (auto cuVS)
- highest centroid-adjacency recall: cuML UMAP (0.75)
- highest balanced context accuracy: cuML UMAP (0.320419)
- lowest diameter-normalized beta-0 bottleneck distance: best observed value 0.35112; practically comparable within 5%: cuML t-SNE, DiRe (spectral init), DiRe (auto cuVS), DiRe (IVF-Flat control), DiRe (topology preset), cuML UMAP
- lowest diameter-normalized beta-1 bottleneck distance: best observed value 0.0245941; practically comparable within 5%: DiRe (topology preset), cuML UMAP, cuML t-SNE, DiRe (spectral init), DiRe (IVF-Flat control)
- lowest beta-0 DTW: DiRe (auto cuVS) (3.46995)
- lowest beta-1 DTW: DiRe (topology preset) (0.11005)
