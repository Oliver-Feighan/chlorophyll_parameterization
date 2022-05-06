%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_151_chromophore_5 ZINDO

0 1
Mg   24.359   -6.630   45.560
C   26.787   -4.209   44.460
C   22.201   -5.365   43.169
C   22.467   -9.333   45.952
C   27.069   -8.285   47.225
N   24.477   -5.066   44.075
C   25.533   -4.244   43.778
C   25.157   -3.191   42.680
C   23.607   -3.362   42.571
C   23.379   -4.654   43.362
C   22.586   -2.169   42.890
C   25.989   -3.363   41.332
C   26.598   -2.090   40.749
C   26.329   -1.918   39.169
O   25.531   -1.141   38.614
O   26.951   -2.907   38.443
N   22.518   -7.278   44.736
C   21.821   -6.606   43.790
C   20.550   -7.319   43.534
C   20.597   -8.494   44.299
C   21.923   -8.493   44.966
C   19.459   -6.924   42.538
C   19.580   -9.673   44.238
O   18.554   -9.508   43.628
C   19.787   -11.036   44.935
N   24.711   -8.558   46.427
C   23.619   -9.443   46.682
C   24.204   -10.770   47.328
C   25.679   -10.394   47.693
C   25.845   -8.997   47.066
C   23.270   -11.230   48.523
C   26.774   -11.373   47.182
C   27.020   -11.273   45.670
N   26.486   -6.299   45.902
C   27.426   -7.025   46.590
C   28.753   -6.446   46.466
C   28.492   -5.325   45.558
C   27.078   -5.275   45.344
C   30.003   -6.759   47.238
C   29.132   -4.183   44.922
O   30.275   -3.745   45.023
C   28.044   -3.439   44.106
C   28.066   -2.013   44.470
O   28.424   -1.179   43.631
O   27.683   -1.786   45.800
C   27.795   -0.379   46.154
C   26.676   -3.136   36.998
C   26.111   -4.457   36.783
C   26.636   -5.510   36.170
C   28.019   -5.501   35.411
C   25.927   -6.814   36.401
C   24.669   -7.077   35.541
C   25.045   -7.817   34.230
C   24.545   -7.093   32.951
C   23.842   -8.152   31.990
C   25.718   -6.366   32.181
C   25.373   -4.840   32.169
C   26.371   -4.056   31.355
C   25.857   -3.617   29.977
C   25.121   -2.319   30.166
C   26.920   -3.623   28.835
C   26.607   -4.566   27.629
C   26.056   -3.698   26.462
C   24.972   -4.522   25.636
C   25.664   -5.583   24.701
C   24.031   -3.567   24.900
H   21.524   -5.015   42.387
H   21.746   -10.098   46.244
H   27.894   -8.786   47.733
H   25.352   -2.221   43.135
H   23.396   -3.544   41.518
H   23.114   -1.432   43.496
H   21.710   -2.564   43.404
H   22.285   -1.650   41.979
H   25.294   -3.590   40.524
H   26.719   -4.159   41.474
H   27.652   -2.238   40.986
H   26.173   -1.211   41.233
H   19.670   -5.965   42.064
H   18.566   -6.742   43.136
H   19.211   -7.687   41.801
H   19.585   -10.855   45.991
H   20.812   -11.368   44.770
H   19.138   -11.854   44.626
H   24.166   -11.499   46.519
H   25.832   -10.199   48.754
H   22.887   -12.229   48.316
H   22.478   -10.518   48.754
H   23.900   -11.449   49.385
H   26.294   -12.348   47.258
H   27.704   -11.276   47.742
H   26.905   -12.260   45.221
H   28.029   -10.943   45.423
H   26.263   -10.654   45.188
H   30.855   -6.714   46.560
H   29.833   -7.789   47.551
H   30.100   -6.096   48.098
H   28.114   -3.525   43.022
H   27.909   0.225   45.254
H   28.733   -0.225   46.688
H   27.026   -0.037   46.846
H   27.678   -3.079   36.574
H   25.999   -2.370   36.620
H   25.246   -4.689   37.404
H   28.424   -6.511   35.486
H   28.709   -4.714   35.714
H   27.886   -5.288   34.351
H   25.637   -6.807   37.452
H   26.581   -7.655   36.171
H   24.109   -6.170   35.315
H   24.107   -7.754   36.184
H   24.761   -8.858   34.387
H   26.116   -7.894   34.041
H   23.692   -6.470   33.220
H   24.230   -7.918   30.999
H   22.752   -8.152   31.968
H   24.104   -9.181   32.238
H   25.886   -6.612   31.133
H   26.673   -6.509   32.687
H   25.447   -4.550   33.217
H   24.344   -4.662   31.857
H   27.294   -4.631   31.291
H   26.573   -3.236   32.045
H   25.055   -4.283   29.656
H   25.878   -1.554   29.989
H   24.694   -2.326   31.169
H   24.236   -2.219   29.537
H   27.812   -4.057   29.286
H   27.086   -2.620   28.442
H   25.939   -5.306   28.069
H   27.456   -5.044   27.141
H   26.827   -3.504   25.716
H   25.654   -2.733   26.771
H   24.313   -5.113   26.272
H   25.437   -6.596   25.035
H   26.754   -5.619   24.687
H   25.322   -5.399   23.682
H   23.012   -3.946   24.985
H   24.329   -3.404   23.864
H   23.877   -2.631   25.437

