%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_201_chromophore_16 TDDFT with blyp functional

0 1
Mg   40.116   41.253   27.492
C   39.232   43.658   29.908
C   41.125   39.180   30.109
C   41.102   39.220   25.185
C   39.342   43.746   25.084
N   40.121   41.380   29.741
C   39.638   42.433   30.478
C   39.723   42.130   32.016
C   40.375   40.701   32.061
C   40.500   40.352   30.568
C   41.656   40.750   32.922
C   38.363   42.149   32.799
C   38.217   41.272   34.027
C   37.328   41.868   35.151
O   36.103   41.870   35.223
O   38.078   42.608   35.989
N   41.005   39.386   27.650
C   41.313   38.733   28.773
C   42.009   37.468   28.387
C   42.008   37.418   26.921
C   41.351   38.694   26.509
C   42.342   36.422   29.447
C   42.630   36.392   25.979
O   42.565   36.530   24.738
C   43.358   35.228   26.594
N   40.166   41.441   25.404
C   40.656   40.441   24.660
C   40.544   40.764   23.213
C   39.621   42.040   23.218
C   39.622   42.434   24.676
C   41.901   40.934   22.392
C   38.207   41.824   22.709
C   37.165   40.872   23.495
N   39.566   43.279   27.490
C   39.432   44.249   26.427
C   39.213   45.541   26.957
C   39.061   45.286   28.293
C   39.265   43.969   28.566
C   38.973   46.840   26.209
C   38.670   46.061   29.550
O   38.353   47.217   29.754
C   38.766   44.944   30.674
C   39.740   45.385   31.617
O   40.954   45.545   31.336
O   39.144   45.846   32.787
C   40.004   46.699   33.678
C   37.431   43.612   36.842
C   37.086   43.097   38.243
C   36.562   43.685   39.337
C   35.986   45.098   39.240
C   36.343   42.734   40.518
C   37.521   42.070   41.167
C   37.454   40.536   41.229
C   38.118   39.898   42.447
C   39.079   38.680   42.060
C   37.070   39.299   43.377
C   37.317   39.760   44.859
C   37.412   38.592   45.884
C   36.937   38.910   47.373
C   37.829   39.939   48.084
C   36.768   37.643   48.270
C   35.278   37.291   48.488
C   34.682   36.287   47.507
C   33.119   36.583   47.204
C   32.894   37.380   45.900
C   32.288   35.270   47.213
H   41.364   38.464   30.898
H   41.434   38.502   24.433
H   38.981   44.323   24.230
H   40.379   42.873   32.470
H   39.685   39.963   32.471
H   41.481   40.200   33.847
H   42.005   41.734   33.235
H   42.462   40.244   32.391
H   37.671   41.892   31.998
H   38.107   43.153   33.137
H   39.237   41.133   34.385
H   37.853   40.309   33.667
H   43.412   36.236   29.547
H   41.824   35.522   29.117
H   42.077   36.595   30.490
H   44.263   35.482   27.146
H   43.590   34.630   25.713
H   42.645   34.623   27.155
H   39.942   39.980   22.755
H   40.201   42.838   22.756
H   41.950   40.178   21.609
H   42.666   40.789   23.156
H   42.026   41.953   22.025
H   38.304   41.447   21.691
H   37.726   42.802   22.683
H   36.931   39.991   22.897
H   36.220   41.401   23.617
H   37.694   40.630   24.417
H   39.035   46.823   25.121
H   39.675   47.599   26.555
H   37.969   47.170   26.475
H   37.831   44.749   31.200
H   40.748   47.320   33.178
H   40.578   46.045   34.334
H   39.337   47.254   34.338
H   36.501   43.999   36.425
H   38.017   44.531   36.868
H   37.404   42.062   38.363
H   36.104   45.607   40.197
H   34.900   45.042   39.306
H   36.414   45.709   38.444
H   35.488   42.100   40.285
H   35.951   43.374   41.309
H   37.731   42.511   42.142
H   38.285   42.254   40.412
H   38.047   40.187   40.384
H   36.421   40.201   41.127
H   38.744   40.666   42.903
H   39.723   38.982   41.235
H   38.474   37.917   41.568
H   39.657   38.252   42.879
H   37.031   38.212   43.303
H   36.078   39.735   43.260
H   36.510   40.433   45.148
H   38.242   40.335   44.874
H   38.475   38.382   45.995
H   36.887   37.727   45.478
H   35.972   39.399   47.244
H   37.229   40.783   48.424
H   38.548   40.374   47.390
H   38.426   39.610   48.935
H   37.238   37.736   49.249
H   37.329   36.839   47.794
H   34.725   38.228   48.557
H   35.214   36.962   49.525
H   34.796   35.271   47.885
H   35.270   36.406   46.597
H   32.635   37.165   47.988
H   32.362   38.299   46.145
H   32.265   36.878   45.164
H   33.789   37.510   45.292
H   31.771   35.169   48.168
H   32.844   34.355   47.009
H   31.541   35.311   46.420

