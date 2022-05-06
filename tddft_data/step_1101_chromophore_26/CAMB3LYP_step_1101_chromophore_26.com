%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1101_chromophore_26 TDDFT with cam-b3lyp functional

0 1
Mg   -9.316   18.656   42.937
C   -5.902   17.798   43.083
C   -8.482   21.978   42.602
C   -12.545   19.327   42.386
C   -10.047   15.145   42.766
N   -7.361   19.700   42.636
C   -6.038   19.206   42.825
C   -5.018   20.386   42.813
C   -5.963   21.675   42.978
C   -7.378   21.079   42.619
C   -5.885   22.552   44.319
C   -4.143   20.197   41.552
C   -4.710   19.527   40.278
C   -4.354   20.085   38.915
O   -3.267   19.947   38.424
O   -5.485   20.682   38.228
N   -10.404   20.415   42.687
C   -9.829   21.635   42.578
C   -10.906   22.577   42.261
C   -12.157   21.886   42.182
C   -11.683   20.476   42.422
C   -10.802   24.072   42.049
C   -13.569   22.499   42.087
O   -13.794   23.732   41.947
C   -14.926   21.662   42.166
N   -10.966   17.399   42.465
C   -12.219   17.958   42.465
C   -13.315   16.901   42.342
C   -12.500   15.621   42.571
C   -11.099   16.072   42.561
C   -14.481   17.005   43.346
C   -12.763   14.438   41.557
C   -13.481   13.155   42.164
N   -8.284   16.798   43.032
C   -8.713   15.483   43.018
C   -7.582   14.583   43.320
C   -6.528   15.539   43.396
C   -6.908   16.796   43.139
C   -7.639   13.130   43.587
C   -5.076   15.618   43.619
O   -4.258   14.693   43.930
C   -4.658   17.079   43.425
C   -3.798   17.422   44.651
O   -2.577   17.432   44.705
O   -4.661   17.631   45.698
C   -3.988   18.012   46.983
C   -5.228   21.315   36.870
C   -6.154   20.642   35.786
C   -6.034   19.349   35.387
C   -4.992   18.283   35.725
C   -7.046   18.890   34.352
C   -8.428   18.358   34.937
C   -9.395   18.325   33.672
C   -10.475   19.483   33.918
C   -11.777   18.847   34.139
C   -10.502   20.553   32.766
C   -11.010   21.959   33.158
C   -9.858   23.027   33.151
C   -10.254   24.264   32.288
C   -9.851   25.628   32.889
C   -9.685   24.185   30.835
C   -10.743   24.340   29.752
C   -10.880   23.247   28.636
C   -12.352   23.061   28.195
C   -12.471   21.618   27.682
C   -12.807   24.137   27.161
H   -8.227   23.040   42.587
H   -13.622   19.446   42.248
H   -10.182   14.061   42.773
H   -4.341   20.327   43.665
H   -5.628   22.391   42.227
H   -5.016   22.434   44.967
H   -6.846   22.479   44.829
H   -5.915   23.589   43.986
H   -3.189   19.790   41.887
H   -3.822   21.165   41.166
H   -5.788   19.692   40.287
H   -4.441   18.473   40.214
H   -10.966   24.406   41.024
H   -9.790   24.310   42.377
H   -11.638   24.526   42.581
H   -14.938   20.995   43.028
H   -15.229   21.190   41.231
H   -15.736   22.375   42.319
H   -13.789   16.847   41.362
H   -12.690   15.255   43.580
H   -14.093   17.751   44.040
H   -14.552   16.054   43.873
H   -15.351   17.363   42.797
H   -11.814   14.126   41.120
H   -13.351   14.724   40.685
H   -13.877   13.251   43.175
H   -12.682   12.419   42.251
H   -14.275   12.766   41.525
H   -6.614   12.813   43.779
H   -7.926   12.574   42.695
H   -8.172   12.861   44.499
H   -4.102   17.121   42.488
H   -4.656   18.708   47.490
H   -2.993   18.444   46.872
H   -3.914   17.175   47.677
H   -4.182   21.159   36.606
H   -5.510   22.366   36.937
H   -7.021   21.234   35.493
H   -4.358   18.737   36.486
H   -4.364   18.000   34.880
H   -5.442   17.382   36.144
H   -6.514   18.053   33.900
H   -7.076   19.570   33.500
H   -8.792   18.876   35.825
H   -8.201   17.311   35.137
H   -9.878   17.349   33.631
H   -8.842   18.519   32.753
H   -10.206   20.012   34.832
H   -12.098   18.872   35.180
H   -11.800   17.843   33.717
H   -12.649   19.399   33.787
H   -11.173   20.178   31.993
H   -9.455   20.636   32.475
H   -11.445   21.911   34.156
H   -11.843   22.264   32.525
H   -8.949   22.569   32.763
H   -9.660   23.375   34.165
H   -11.338   24.366   32.232
H   -8.899   25.493   33.403
H   -10.519   26.024   33.655
H   -9.699   26.383   32.117
H   -9.246   23.207   30.638
H   -8.892   24.918   30.685
H   -10.411   25.287   29.326
H   -11.757   24.425   30.143
H   -10.469   22.297   28.977
H   -10.252   23.548   27.798
H   -12.993   23.141   29.073
H   -11.534   21.062   27.694
H   -12.895   21.560   26.680
H   -13.196   21.213   28.389
H   -13.585   24.765   27.597
H   -13.252   23.723   26.257
H   -11.980   24.777   26.854

