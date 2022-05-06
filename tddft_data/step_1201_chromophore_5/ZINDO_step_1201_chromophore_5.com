%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1201_chromophore_5 ZINDO

0 1
Mg   25.310   -8.057   46.857
C   26.997   -5.367   45.333
C   22.674   -7.417   44.603
C   23.843   -10.939   47.674
C   28.218   -9.021   48.427
N   25.012   -6.732   45.066
C   25.756   -5.655   44.671
C   25.225   -4.977   43.432
C   23.773   -5.530   43.344
C   23.831   -6.688   44.357
C   22.694   -4.538   43.608
C   26.082   -5.258   42.064
C   26.197   -3.920   41.232
C   25.056   -3.784   40.261
O   23.903   -3.398   40.519
O   25.430   -4.318   39.064
N   23.403   -8.944   46.355
C   22.503   -8.546   45.447
C   21.336   -9.439   45.447
C   21.672   -10.489   46.277
C   23.016   -10.160   46.811
C   20.066   -9.148   44.730
C   20.784   -11.791   46.535
O   19.695   -11.789   46.050
C   21.270   -12.989   47.164
N   26.004   -9.758   47.776
C   25.195   -10.775   48.154
C   25.873   -11.779   49.080
C   27.223   -11.150   49.312
C   27.179   -9.908   48.429
C   25.108   -12.017   50.473
C   28.398   -12.121   49.233
C   29.258   -12.278   50.508
N   27.270   -7.451   46.824
C   28.291   -7.792   47.682
C   29.385   -6.851   47.559
C   28.938   -5.902   46.572
C   27.591   -6.283   46.234
C   30.609   -6.907   48.292
C   29.320   -4.697   45.907
O   30.413   -4.153   45.782
C   28.035   -4.314   45.027
C   27.485   -3.046   45.460
O   27.213   -2.163   44.656
O   27.219   -2.855   46.797
C   26.637   -1.554   47.167
C   24.767   -3.843   37.812
C   23.924   -4.876   37.172
C   24.243   -6.049   36.477
C   25.644   -6.601   36.165
C   23.181   -6.832   35.802
C   22.867   -6.408   34.297
C   23.026   -7.491   33.243
C   23.672   -6.992   31.989
C   22.839   -7.565   30.860
C   25.152   -7.388   31.806
C   26.165   -6.220   31.402
C   26.576   -6.265   29.925
C   26.166   -5.050   29.044
C   27.235   -3.993   29.103
C   25.740   -5.496   27.683
C   24.209   -5.484   27.497
C   23.628   -4.205   26.742
C   22.158   -3.934   27.115
C   21.492   -2.999   26.172
C   22.120   -3.160   28.521
H   21.812   -7.091   44.018
H   23.480   -11.837   48.178
H   29.073   -9.253   49.067
H   25.274   -3.911   43.654
H   23.676   -5.907   42.326
H   22.305   -4.169   42.660
H   23.026   -3.594   44.040
H   21.822   -4.892   44.158
H   25.586   -6.009   41.448
H   27.065   -5.601   42.385
H   27.175   -3.879   40.752
H   26.032   -3.005   41.799
H   20.332   -9.439   43.714
H   19.903   -8.071   44.775
H   19.123   -9.545   45.104
H   21.891   -12.985   48.059
H   21.931   -13.384   46.392
H   20.461   -13.711   47.278
H   25.855   -12.732   48.551
H   27.301   -10.743   50.320
H   24.234   -11.366   50.474
H   25.707   -11.776   51.352
H   24.915   -13.077   50.641
H   29.021   -11.738   48.424
H   27.986   -13.120   49.089
H   30.304   -12.263   50.201
H   28.783   -13.088   51.062
H   29.156   -11.444   51.203
H   31.187   -5.999   48.125
H   31.239   -7.796   48.241
H   30.259   -6.832   49.322
H   28.357   -4.284   43.986
H   26.951   -1.269   48.171
H   25.563   -1.672   47.314
H   26.820   -0.713   46.498
H   25.650   -3.546   37.245
H   24.182   -2.936   37.967
H   22.872   -4.589   37.149
H   26.444   -5.870   36.281
H   25.643   -6.838   35.101
H   25.829   -7.518   36.724
H   22.217   -6.770   36.306
H   23.312   -7.913   35.833
H   23.545   -5.581   34.086
H   21.841   -6.058   34.188
H   22.097   -8.056   33.169
H   23.675   -8.276   33.633
H   23.609   -5.904   31.962
H   22.587   -8.605   31.070
H   23.378   -7.625   29.915
H   21.906   -7.037   30.660
H   25.092   -8.267   31.164
H   25.520   -7.695   32.785
H   27.078   -6.364   31.979
H   25.837   -5.270   31.824
H   26.274   -7.190   29.432
H   27.665   -6.295   29.960
H   25.299   -4.499   29.410
H   27.291   -3.375   29.999
H   27.049   -3.251   28.326
H   28.204   -4.399   28.814
H   26.172   -6.402   27.258
H   26.151   -4.719   27.039
H   23.717   -5.643   28.456
H   24.080   -6.343   26.838
H   23.532   -4.465   25.688
H   24.231   -3.306   26.866
H   21.541   -4.826   27.225
H   22.190   -2.242   25.812
H   20.717   -2.365   26.601
H   21.092   -3.601   25.356
H   21.314   -3.590   29.115
H   22.000   -2.098   28.305
H   23.078   -3.353   29.005

