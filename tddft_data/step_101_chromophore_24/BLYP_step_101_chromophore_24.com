%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_101_chromophore_24 TDDFT with blyp functional

0 1
Mg   -0.282   43.664   24.385
C   1.462   43.215   27.360
C   -3.216   43.004   26.033
C   -1.873   43.350   21.309
C   2.776   43.882   22.616
N   -0.815   43.314   26.503
C   0.104   43.140   27.519
C   -0.604   42.788   28.751
C   -2.135   42.638   28.297
C   -2.076   42.955   26.850
C   -2.676   41.232   28.614
C   -0.426   43.802   29.976
C   -0.397   43.221   31.441
C   -0.718   41.766   31.755
O   -0.009   40.771   31.653
O   -1.986   41.745   32.335
N   -2.280   43.374   23.825
C   -3.355   43.190   24.630
C   -4.551   43.262   23.915
C   -4.173   43.223   22.536
C   -2.717   43.322   22.485
C   -5.947   43.201   24.549
C   -5.093   43.128   21.345
O   -4.688   43.270   20.224
C   -6.566   42.646   21.526
N   0.311   43.594   22.358
C   -0.472   43.440   21.259
C   0.250   43.405   19.914
C   1.635   43.717   20.333
C   1.552   43.685   21.887
C   0.050   42.118   19.029
C   2.266   45.011   19.760
C   1.505   46.371   19.862
N   1.795   43.520   24.828
C   2.891   43.698   23.994
C   4.102   43.634   24.871
C   3.585   43.456   26.169
C   2.167   43.388   26.116
C   5.622   43.644   24.439
C   3.930   43.284   27.545
O   4.983   43.185   28.105
C   2.589   43.068   28.350
C   2.717   41.878   29.168
O   2.653   40.751   28.708
O   3.009   42.232   30.448
C   3.397   41.192   31.344
C   -2.504   40.405   32.638
C   -3.429   40.594   33.777
C   -3.119   41.080   34.945
C   -1.718   41.643   35.379
C   -4.169   41.004   36.058
C   -4.876   42.361   36.415
C   -6.174   42.477   35.560
C   -7.416   42.664   36.401
C   -8.096   43.972   36.061
C   -8.399   41.487   36.327
C   -8.803   41.091   37.753
C   -9.861   39.996   37.676
C   -10.925   40.099   38.771
C   -11.465   38.646   39.129
C   -12.095   41.010   38.301
C   -12.290   42.177   39.174
C   -11.729   43.533   38.571
C   -12.832   44.496   38.300
C   -12.403   45.432   37.180
C   -13.249   45.329   39.541
H   -4.134   42.896   26.614
H   -2.382   43.152   20.364
H   3.641   43.923   21.951
H   -0.226   41.861   29.183
H   -2.856   43.333   28.729
H   -1.941   40.429   28.658
H   -3.402   40.927   27.860
H   -3.069   41.272   29.630
H   -1.297   44.456   29.948
H   0.382   44.495   29.740
H   -1.056   43.861   32.027
H   0.645   43.348   31.734
H   -6.433   42.245   24.355
H   -6.544   43.980   24.075
H   -6.078   43.465   25.599
H   -7.220   43.478   21.787
H   -6.638   41.760   22.157
H   -6.825   42.112   20.612
H   -0.155   44.203   19.292
H   2.352   42.954   20.031
H   0.728   41.397   19.485
H   0.218   42.348   17.977
H   -0.960   41.731   19.169
H   2.426   44.938   18.684
H   3.234   45.199   20.225
H   0.932   46.830   19.056
H   2.155   47.227   20.047
H   0.833   46.442   20.718
H   6.053   44.546   24.873
H   5.721   43.628   23.354
H   6.138   42.739   24.759
H   2.468   43.914   29.026
H   4.165   40.600   30.847
H   2.547   40.532   31.515
H   3.769   41.639   32.266
H   -1.731   39.764   33.065
H   -3.000   39.969   31.771
H   -4.438   40.191   33.689
H   -0.935   41.404   34.659
H   -1.470   41.230   36.357
H   -1.782   42.729   35.448
H   -3.673   40.693   36.977
H   -4.875   40.202   35.843
H   -4.268   43.204   36.088
H   -5.036   42.590   37.469
H   -6.329   41.593   34.942
H   -5.994   43.249   34.812
H   -7.066   42.804   37.424
H   -7.313   44.684   35.801
H   -8.582   44.325   36.971
H   -8.880   43.860   35.312
H   -8.091   40.540   35.883
H   -9.297   41.713   35.751
H   -9.096   41.895   38.427
H   -7.944   40.546   38.145
H   -9.371   39.025   37.594
H   -10.318   39.965   36.686
H   -10.498   40.501   39.689
H   -12.003   38.619   40.076
H   -10.699   37.887   38.967
H   -12.151   38.426   38.311
H   -13.041   40.469   38.336
H   -12.008   41.462   37.313
H   -11.914   42.059   40.190
H   -13.315   42.407   39.465
H   -11.070   43.329   37.727
H   -11.092   43.895   39.377
H   -13.761   43.954   38.120
H   -11.372   45.768   37.298
H   -13.061   46.300   37.135
H   -12.579   44.954   36.217
H   -14.146   44.840   39.920
H   -13.513   46.370   39.356
H   -12.606   45.294   40.421

