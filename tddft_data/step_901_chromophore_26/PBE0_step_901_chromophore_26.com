%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_901_chromophore_26 TDDFT with PBE1PBE functional

0 1
Mg   -8.989   19.048   42.666
C   -5.622   18.709   42.857
C   -8.609   22.398   41.897
C   -12.406   19.322   42.255
C   -9.283   15.555   42.546
N   -7.300   20.348   42.393
C   -5.994   20.019   42.518
C   -5.066   21.206   42.250
C   -6.093   22.347   42.117
C   -7.452   21.658   42.161
C   -5.955   23.544   43.145
C   -4.216   20.891   40.982
C   -4.787   20.062   39.747
C   -4.334   20.517   38.353
O   -3.286   21.142   38.144
O   -5.164   19.981   37.396
N   -10.381   20.666   42.055
C   -9.976   21.966   41.860
C   -11.167   22.765   41.668
C   -12.336   21.894   41.914
C   -11.720   20.534   42.062
C   -11.172   24.166   41.133
C   -13.728   22.368   41.849
O   -13.961   23.551   41.616
C   -14.997   21.444   41.966
N   -10.598   17.631   42.159
C   -11.887   18.008   42.277
C   -12.849   16.795   42.370
C   -11.874   15.555   42.107
C   -10.470   16.237   42.245
C   -13.546   16.740   43.710
C   -12.100   14.798   40.699
C   -12.197   13.292   40.730
N   -7.769   17.467   42.744
C   -7.956   16.093   42.748
C   -6.811   15.373   43.023
C   -5.802   16.404   43.051
C   -6.460   17.615   42.829
C   -6.705   13.908   43.075
C   -4.407   16.678   43.286
O   -3.459   15.926   43.613
C   -4.222   18.162   43.193
C   -3.704   18.808   44.380
O   -2.685   19.536   44.403
O   -4.489   18.310   45.431
C   -4.010   18.675   46.841
C   -4.698   20.393   36.102
C   -5.721   20.061   35.034
C   -6.098   18.801   34.645
C   -5.435   17.479   35.145
C   -7.298   18.764   33.651
C   -8.672   18.891   34.270
C   -9.857   18.506   33.346
C   -10.917   19.563   33.066
C   -12.265   18.906   32.926
C   -10.716   20.629   31.913
C   -10.889   22.131   32.323
C   -9.753   23.074   31.820
C   -10.222   24.132   30.813
C   -9.508   25.421   31.003
C   -10.117   23.666   29.271
C   -10.962   24.545   28.240
C   -10.261   24.634   26.849
C   -11.095   24.033   25.703
C   -10.171   23.278   24.621
C   -11.737   25.175   24.965
H   -8.531   23.486   41.840
H   -13.496   19.385   42.274
H   -9.344   14.480   42.727
H   -4.431   21.208   43.137
H   -6.040   22.915   41.187
H   -5.856   24.469   42.576
H   -5.061   23.440   43.760
H   -6.826   23.551   43.801
H   -3.260   20.437   41.241
H   -3.857   21.836   40.575
H   -5.870   20.136   39.659
H   -4.629   18.985   39.810
H   -11.761   24.206   40.217
H   -10.115   24.394   40.997
H   -11.598   24.771   41.933
H   -14.980   20.908   42.915
H   -14.966   20.604   41.272
H   -15.856   22.095   41.801
H   -13.611   16.961   41.609
H   -11.904   14.753   42.844
H   -14.564   16.377   43.567
H   -13.595   17.707   44.211
H   -13.013   16.173   44.473
H   -11.251   15.078   40.076
H   -12.989   15.103   40.146
H   -12.987   12.862   40.115
H   -12.148   12.838   41.720
H   -11.398   12.901   40.100
H   -5.724   13.568   42.743
H   -7.334   13.501   42.284
H   -6.970   13.369   43.985
H   -3.541   18.468   42.399
H   -2.967   18.973   46.733
H   -4.043   17.805   47.496
H   -4.482   19.596   47.184
H   -3.805   19.795   35.923
H   -4.515   21.467   36.076
H   -6.211   20.885   34.516
H   -4.838   16.950   34.402
H   -6.247   16.765   35.285
H   -4.742   17.535   35.985
H   -7.301   17.755   33.239
H   -7.178   19.443   32.806
H   -8.829   19.931   34.556
H   -8.717   18.287   35.176
H   -10.350   17.614   33.734
H   -9.502   18.228   32.354
H   -10.958   20.173   33.968
H   -12.654   18.704   33.925
H   -12.087   17.946   32.443
H   -13.118   19.467   32.544
H   -11.436   20.311   31.159
H   -9.729   20.481   31.474
H   -10.861   22.204   33.410
H   -11.879   22.456   32.004
H   -8.887   22.593   31.366
H   -9.363   23.618   32.681
H   -11.291   24.349   30.830
H   -9.015   25.630   31.953
H   -10.246   26.213   30.876
H   -8.775   25.571   30.211
H   -10.504   22.648   29.232
H   -9.093   23.505   28.934
H   -10.953   25.502   28.761
H   -11.941   24.071   28.158
H   -9.258   24.230   26.988
H   -10.058   25.699   26.743
H   -11.873   23.314   25.962
H   -9.786   22.370   25.087
H   -9.339   23.914   24.321
H   -10.809   23.171   23.744
H   -11.429   25.324   23.930
H   -11.573   26.146   25.433
H   -12.824   25.094   24.941

