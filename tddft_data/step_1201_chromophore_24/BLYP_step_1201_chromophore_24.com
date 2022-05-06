%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1201_chromophore_24 TDDFT with blyp functional

0 1
Mg   -0.082   43.948   24.845
C   1.978   43.246   27.544
C   -2.554   42.789   26.599
C   -1.888   43.784   21.939
C   2.729   44.735   22.907
N   -0.230   43.324   26.872
C   0.696   43.187   27.873
C   0.108   42.860   29.197
C   -1.399   42.526   28.798
C   -1.407   42.825   27.310
C   -1.782   40.989   29.126
C   0.187   44.025   30.319
C   0.948   43.733   31.583
C   0.903   42.251   32.233
O   1.829   41.477   32.285
O   -0.391   42.031   32.643
N   -1.981   43.476   24.393
C   -2.894   42.997   25.209
C   -4.186   42.745   24.512
C   -4.035   43.195   23.217
C   -2.598   43.548   23.119
C   -5.384   42.283   25.283
C   -5.054   43.307   22.050
O   -4.785   43.745   20.912
C   -6.483   42.810   22.362
N   0.336   44.140   22.679
C   -0.546   44.113   21.692
C   0.062   44.635   20.361
C   1.441   45.122   20.785
C   1.535   44.713   22.263
C   0.070   43.579   19.190
C   1.630   46.678   20.569
C   0.629   47.598   21.448
N   1.958   43.990   25.140
C   2.991   44.365   24.267
C   4.232   44.332   24.937
C   3.914   43.859   26.180
C   2.493   43.696   26.304
C   5.582   44.797   24.426
C   4.450   43.491   27.455
O   5.579   43.540   27.917
C   3.259   43.039   28.435
C   3.513   41.685   28.884
O   3.153   40.673   28.298
O   4.012   41.696   30.137
C   4.360   40.452   30.789
C   -0.644   40.848   33.443
C   -1.907   41.056   34.296
C   -2.198   40.588   35.484
C   -1.342   39.554   36.177
C   -3.574   40.846   36.127
C   -3.461   42.146   36.966
C   -3.954   43.370   36.091
C   -4.905   44.203   36.969
C   -4.326   45.571   37.385
C   -6.274   44.277   36.165
C   -7.451   43.841   37.100
C   -8.791   43.581   36.323
C   -9.394   42.183   36.281
C   -9.687   41.736   34.786
C   -10.628   42.057   37.246
C   -10.580   40.835   38.211
C   -11.960   40.146   38.381
C   -11.931   38.690   38.928
C   -11.973   37.728   37.687
C   -13.038   38.267   39.900
H   -3.418   42.396   27.139
H   -2.489   43.810   21.028
H   3.542   45.161   22.315
H   0.684   41.979   29.480
H   -2.170   43.178   29.210
H   -0.950   40.498   29.631
H   -2.140   40.348   28.321
H   -2.644   40.988   29.793
H   -0.812   44.296   30.660
H   0.648   44.895   29.852
H   0.689   44.502   32.310
H   2.005   43.929   31.399
H   -5.880   43.211   25.568
H   -5.171   41.803   26.237
H   -6.003   41.518   24.815
H   -6.809   43.420   23.205
H   -6.454   41.735   22.538
H   -7.247   42.905   21.590
H   -0.520   45.507   20.063
H   2.135   44.617   20.113
H   0.579   44.085   18.369
H   -0.911   43.173   18.944
H   0.785   42.870   19.608
H   1.486   46.836   19.500
H   2.672   46.856   20.835
H   0.116   48.179   20.682
H   1.217   48.319   22.016
H   -0.127   47.006   21.965
H   6.227   45.204   25.205
H   5.418   45.518   23.626
H   6.030   43.949   23.907
H   3.240   43.733   29.276
H   3.678   39.655   30.495
H   4.325   40.562   31.873
H   5.309   40.043   30.442
H   0.212   40.718   34.105
H   -0.797   39.957   32.834
H   -2.682   41.676   33.846
H   -1.600   39.481   37.233
H   -0.279   39.778   36.096
H   -1.626   38.656   35.628
H   -3.785   39.959   36.724
H   -4.344   40.879   35.356
H   -2.439   42.378   37.267
H   -4.080   41.941   37.840
H   -4.319   43.068   35.109
H   -3.153   43.997   35.700
H   -4.981   43.718   37.943
H   -3.286   45.647   37.068
H   -4.344   45.725   38.464
H   -4.858   46.394   36.909
H   -6.303   43.582   35.327
H   -6.393   45.307   35.827
H   -7.660   44.766   37.636
H   -7.127   43.044   37.770
H   -8.721   44.030   35.332
H   -9.451   44.229   36.901
H   -8.611   41.471   36.542
H   -9.188   42.399   34.079
H   -10.731   41.576   34.517
H   -9.201   40.772   34.635
H   -11.529   41.958   36.640
H   -10.882   42.929   37.849
H   -10.270   41.131   39.213
H   -9.887   40.117   37.773
H   -12.433   40.283   37.409
H   -12.433   40.818   39.097
H   -10.989   38.423   39.408
H   -13.004   37.480   37.435
H   -11.380   36.822   37.811
H   -11.513   38.264   36.857
H   -12.570   37.912   40.818
H   -13.558   37.364   39.581
H   -13.587   39.196   40.054

