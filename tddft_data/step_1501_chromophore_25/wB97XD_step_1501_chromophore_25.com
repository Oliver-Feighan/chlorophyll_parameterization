%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1501_chromophore_25 TDDFT with wB97XD functional

0 1
Mg   -2.019   34.138   26.358
C   -3.311   32.493   29.211
C   -0.781   36.552   28.426
C   -1.434   36.107   23.557
C   -4.029   32.017   24.367
N   -1.942   34.437   28.671
C   -2.502   33.602   29.597
C   -2.149   34.113   31.014
C   -1.469   35.474   30.667
C   -1.342   35.423   29.152
C   -1.995   36.757   31.299
C   -1.267   32.981   31.668
C   -1.895   32.381   32.896
C   -1.177   32.543   34.187
O   -0.299   31.781   34.551
O   -1.593   33.632   34.854
N   -1.116   36.002   26.068
C   -0.657   36.833   27.037
C   -0.081   38.002   26.462
C   -0.148   37.898   25.054
C   -0.897   36.598   24.818
C   0.625   39.086   27.303
C   0.357   38.831   23.945
O   0.148   38.604   22.755
C   1.103   40.061   24.246
N   -2.658   34.049   24.277
C   -2.253   34.961   23.345
C   -2.852   34.589   21.948
C   -3.653   33.290   22.226
C   -3.418   33.079   23.720
C   -3.752   35.670   21.314
C   -3.139   32.152   21.305
C   -2.174   31.135   21.840
N   -3.270   32.500   26.646
C   -3.983   31.688   25.785
C   -4.551   30.619   26.506
C   -4.373   30.925   27.846
C   -3.621   32.103   27.873
C   -5.291   29.453   25.995
C   -4.831   30.586   29.199
O   -5.557   29.733   29.659
C   -4.035   31.536   30.135
C   -5.070   32.297   30.976
O   -5.725   33.257   30.669
O   -5.190   31.516   32.101
C   -5.875   32.210   33.177
C   -0.826   33.907   36.186
C   -1.669   34.424   37.268
C   -1.357   34.688   38.546
C   0.134   34.564   38.958
C   -2.338   34.963   39.712
C   -2.189   36.323   40.397
C   -1.514   36.185   41.862
C   -1.702   37.475   42.726
C   -0.558   38.503   42.485
C   -1.800   37.119   44.238
C   -0.680   36.089   44.757
C   0.189   36.763   45.906
C   1.307   35.755   46.505
C   1.331   35.732   48.045
C   2.798   36.168   45.932
C   2.965   35.190   44.768
C   4.156   35.532   43.841
C   5.319   34.564   43.936
C   6.651   35.423   44.198
C   5.534   33.557   42.770
H   -0.337   37.361   29.008
H   -1.307   36.602   22.593
H   -4.589   31.317   23.743
H   -3.085   34.253   31.556
H   -0.448   35.401   31.042
H   -2.909   36.510   31.840
H   -2.342   37.439   30.522
H   -1.271   37.252   31.945
H   -0.253   33.335   31.856
H   -1.250   32.106   31.019
H   -1.941   31.295   32.815
H   -2.923   32.700   33.069
H   1.694   38.923   27.162
H   0.290   39.164   28.338
H   0.414   39.981   26.718
H   1.557   40.430   23.326
H   1.953   39.883   24.904
H   0.456   40.846   24.636
H   -1.946   34.387   21.377
H   -4.735   33.402   22.171
H   -3.092   36.125   20.574
H   -3.964   36.449   22.046
H   -4.629   35.205   20.865
H   -2.666   32.729   20.510
H   -4.001   31.656   20.858
H   -2.583   30.129   21.940
H   -1.857   31.366   22.857
H   -1.199   31.057   21.360
H   -4.790   28.522   25.731
H   -5.853   29.873   25.160
H   -6.068   29.129   26.688
H   -3.443   30.942   30.831
H   -6.744   32.745   32.793
H   -5.237   32.814   33.821
H   -6.293   31.422   33.805
H   -0.178   34.723   35.867
H   -0.281   33.027   36.526
H   -2.730   34.593   37.088
H   0.257   33.830   39.754
H   0.376   35.507   39.448
H   0.768   34.312   38.108
H   -2.177   34.193   40.466
H   -3.325   35.031   39.254
H   -3.234   36.617   40.498
H   -1.636   36.969   39.715
H   -0.446   35.992   41.765
H   -1.928   35.334   42.403
H   -2.686   37.856   42.454
H   -0.923   39.225   41.754
H   0.303   37.985   42.062
H   -0.266   38.983   43.419
H   -2.688   36.599   44.597
H   -1.683   37.984   44.890
H   -0.069   35.581   44.011
H   -1.187   35.265   45.258
H   -0.584   36.931   46.656
H   0.690   37.694   45.640
H   1.038   34.746   46.194
H   2.183   36.288   48.436
H   1.534   34.740   48.448
H   0.461   36.198   48.508
H   3.603   36.040   46.656
H   2.800   37.180   45.528
H   2.045   35.226   44.185
H   3.062   34.136   45.025
H   4.514   36.532   44.089
H   3.826   35.542   42.802
H   5.231   33.931   44.819
H   6.281   36.021   45.031
H   6.856   35.991   43.291
H   7.451   34.713   44.408
H   4.604   33.269   42.279
H   6.003   32.683   43.221
H   6.269   33.991   42.092

