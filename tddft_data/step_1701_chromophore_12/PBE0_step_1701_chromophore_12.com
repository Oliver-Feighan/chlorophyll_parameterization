%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1701_chromophore_12 TDDFT with PBE1PBE functional

0 1
Mg   47.608   15.977   28.618
C   45.348   15.613   31.307
C   49.334   18.163   30.520
C   49.505   16.708   25.879
C   45.500   13.999   26.594
N   47.360   16.670   30.655
C   46.406   16.434   31.684
C   46.802   17.068   32.970
C   47.889   18.075   32.553
C   48.198   17.628   31.131
C   47.495   19.613   32.690
C   47.264   16.168   34.171
C   47.573   16.983   35.485
C   48.606   16.370   36.478
O   49.264   15.323   36.364
O   48.820   17.294   37.476
N   49.200   17.243   28.283
C   49.812   18.015   29.222
C   50.970   18.797   28.638
C   51.044   18.357   27.248
C   49.889   17.369   27.061
C   51.780   19.803   29.425
C   51.947   18.818   26.099
O   51.964   18.240   25.005
C   52.670   20.060   26.228
N   47.508   15.381   26.578
C   48.436   15.787   25.636
C   48.307   14.992   24.299
C   47.220   13.918   24.678
C   46.741   14.436   26.061
C   47.920   15.969   23.106
C   47.740   12.470   24.640
C   48.889   12.043   23.717
N   45.826   15.074   28.852
C   45.006   14.383   27.927
C   43.774   13.968   28.561
C   43.919   14.368   29.875
C   45.116   15.036   30.026
C   42.626   13.243   28.026
C   43.248   14.411   31.159
O   42.127   14.007   31.495
C   44.197   15.137   32.214
C   43.470   16.193   32.868
O   42.733   16.997   32.303
O   43.860   16.288   34.194
C   43.122   17.360   34.920
C   49.921   17.093   38.301
C   50.033   18.214   39.343
C   51.149   18.670   39.951
C   52.538   18.188   39.775
C   50.979   19.876   40.933
C   51.369   21.235   40.222
C   52.446   22.041   40.984
C   53.371   22.777   39.965
C   54.760   23.002   40.648
C   52.737   24.091   39.537
C   52.282   24.203   38.076
C   53.089   25.192   37.206
C   54.094   24.472   36.185
C   55.376   24.214   36.869
C   54.220   25.182   34.797
C   53.823   24.313   33.675
C   52.320   24.583   33.368
C   51.873   23.665   32.229
C   52.344   24.216   30.846
C   50.330   23.482   32.251
H   49.921   18.892   31.083
H   50.134   16.721   24.986
H   44.951   13.310   25.949
H   46.003   17.679   33.389
H   48.811   17.982   33.127
H   46.448   19.814   32.919
H   47.776   20.248   31.851
H   48.059   19.954   33.558
H   48.246   15.809   33.865
H   46.632   15.325   34.452
H   46.613   17.027   36.000
H   47.954   17.991   35.321
H   51.483   19.977   30.459
H   51.829   20.853   29.137
H   52.824   19.492   29.464
H   52.007   20.905   26.413
H   53.092   20.281   25.247
H   53.561   19.943   26.845
H   49.312   14.626   24.090
H   46.364   14.000   24.008
H   46.959   15.680   22.679
H   48.693   15.937   22.339
H   47.872   17.042   23.292
H   46.904   11.796   24.458
H   48.153   12.251   25.624
H   48.588   11.172   23.134
H   49.760   11.725   24.288
H   49.332   12.783   23.050
H   42.966   12.820   27.081
H   41.815   13.960   27.892
H   42.192   12.520   28.717
H   44.580   14.445   32.964
H   43.556   17.294   35.918
H   42.037   17.320   35.016
H   43.364   18.339   34.505
H   50.795   17.170   37.654
H   49.785   16.135   38.802
H   49.061   18.688   39.480
H   52.936   17.695   40.662
H   53.114   19.090   39.569
H   52.551   17.531   38.905
H   51.606   19.688   41.804
H   49.920   19.905   41.190
H   50.471   21.843   40.331
H   51.598   21.120   39.163
H   53.039   21.433   41.667
H   52.141   22.786   41.718
H   53.578   22.168   39.085
H   54.676   23.057   41.733
H   55.381   23.848   40.355
H   55.390   22.114   40.597
H   53.309   24.953   39.880
H   51.836   24.170   40.145
H   51.244   24.535   38.050
H   52.238   23.199   37.653
H   53.546   25.825   37.967
H   52.379   25.869   36.731
H   53.690   23.461   36.136
H   55.786   24.973   37.535
H   56.101   23.963   36.094
H   55.336   23.278   37.427
H   55.223   25.495   34.506
H   53.624   26.092   34.861
H   54.131   23.277   33.809
H   54.405   24.573   32.790
H   52.258   25.613   33.016
H   51.654   24.503   34.228
H   52.195   22.657   32.491
H   53.245   23.663   30.580
H   52.517   25.291   30.876
H   51.666   23.997   30.020
H   50.135   22.440   32.505
H   49.897   23.643   31.264
H   49.693   24.002   32.966

