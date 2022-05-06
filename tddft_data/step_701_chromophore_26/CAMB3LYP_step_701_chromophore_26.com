%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_701_chromophore_26 TDDFT with cam-b3lyp functional

0 1
Mg   -9.271   18.554   42.942
C   -5.822   18.080   43.142
C   -8.759   21.835   42.287
C   -12.562   18.835   42.291
C   -9.714   14.998   43.002
N   -7.442   19.683   42.439
C   -6.157   19.367   42.772
C   -5.185   20.632   42.517
C   -6.242   21.807   42.494
C   -7.575   21.046   42.357
C   -6.202   22.687   43.821
C   -4.255   20.714   41.191
C   -4.725   20.081   39.792
C   -4.741   21.040   38.612
O   -4.721   22.272   38.643
O   -4.720   20.323   37.489
N   -10.522   20.163   42.399
C   -10.095   21.448   42.231
C   -11.189   22.260   41.947
C   -12.344   21.356   41.816
C   -11.850   20.031   42.191
C   -11.139   23.687   41.686
C   -13.739   21.752   41.407
O   -14.068   22.914   41.265
C   -14.884   20.841   41.333
N   -10.984   17.088   42.641
C   -12.234   17.470   42.483
C   -13.226   16.220   42.653
C   -12.255   14.996   42.831
C   -10.942   15.723   42.860
C   -14.403   16.271   43.683
C   -12.354   13.879   41.757
C   -13.380   12.755   42.062
N   -8.039   16.883   43.023
C   -8.349   15.525   43.159
C   -7.102   14.801   43.402
C   -6.060   15.779   43.412
C   -6.686   17.009   43.129
C   -6.945   13.283   43.505
C   -4.599   15.973   43.473
O   -3.685   15.152   43.569
C   -4.375   17.504   43.323
C   -3.758   17.979   44.627
O   -2.783   18.682   44.690
O   -4.480   17.541   45.667
C   -3.893   17.926   46.948
C   -4.561   20.930   36.234
C   -5.870   20.649   35.546
C   -6.321   19.613   34.778
C   -5.533   18.285   34.790
C   -7.526   19.737   33.895
C   -8.777   18.936   34.338
C   -9.809   18.706   33.163
C   -11.211   19.229   33.502
C   -12.186   18.090   33.927
C   -11.741   20.255   32.434
C   -11.118   21.704   32.498
C   -10.464   22.203   31.110
C   -10.466   23.714   30.882
C   -9.550   24.477   31.796
C   -10.160   23.935   29.437
C   -11.427   24.325   28.618
C   -11.430   23.850   27.161
C   -12.149   22.518   27.016
C   -11.629   21.629   25.888
C   -13.714   22.701   26.943
H   -8.604   22.915   42.342
H   -13.649   18.919   42.234
H   -9.731   13.907   43.053
H   -4.523   20.615   43.382
H   -6.115   22.370   41.569
H   -7.077   22.538   44.454
H   -6.284   23.747   43.582
H   -5.317   22.470   44.419
H   -3.244   20.408   41.458
H   -4.220   21.792   41.032
H   -5.765   19.774   39.906
H   -4.085   19.241   39.524
H   -11.050   23.887   40.618
H   -10.273   24.134   42.175
H   -11.978   24.312   41.990
H   -15.108   20.489   42.340
H   -14.573   19.949   40.788
H   -15.750   21.272   40.831
H   -13.737   16.128   41.695
H   -12.387   14.570   43.826
H   -14.339   17.229   44.199
H   -14.471   15.474   44.424
H   -15.337   16.230   43.123
H   -11.412   13.400   41.488
H   -12.681   14.383   40.848
H   -13.682   12.718   43.109
H   -12.873   11.819   41.829
H   -14.202   12.938   41.370
H   -6.032   13.020   44.040
H   -6.803   12.819   42.529
H   -7.837   12.849   43.957
H   -3.647   17.515   42.512
H   -4.638   17.892   47.743
H   -3.506   18.945   46.960
H   -3.149   17.162   47.174
H   -3.759   20.461   35.664
H   -4.353   21.989   36.386
H   -6.584   21.459   35.402
H   -4.562   18.398   35.271
H   -5.368   17.937   33.770
H   -6.114   17.605   35.413
H   -7.334   19.223   32.953
H   -7.801   20.773   33.699
H   -9.186   19.573   35.122
H   -8.497   17.973   34.764
H   -9.886   17.636   32.970
H   -9.395   19.123   32.245
H   -11.036   19.897   34.346
H   -12.874   18.565   34.626
H   -11.735   17.199   34.363
H   -12.792   17.861   33.051
H   -12.815   20.428   32.491
H   -11.587   19.762   31.474
H   -10.316   21.771   33.233
H   -11.841   22.398   32.925
H   -11.001   21.688   30.313
H   -9.443   21.824   31.145
H   -11.437   24.108   31.181
H   -9.195   23.872   32.631
H   -10.011   25.351   32.256
H   -8.654   24.817   31.276
H   -9.511   23.160   29.028
H   -9.639   24.891   29.395
H   -11.472   25.399   28.439
H   -12.377   24.031   29.063
H   -10.380   23.715   26.901
H   -11.893   24.572   26.488
H   -12.042   21.839   27.861
H   -11.332   20.621   26.177
H   -10.776   22.156   25.460
H   -12.240   21.646   24.985
H   -14.037   22.479   25.926
H   -14.016   23.701   27.257
H   -14.122   22.019   27.689

