%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1701_chromophore_15 TDDFT with blyp functional

0 1
Mg   46.939   34.748   27.732
C   45.472   33.110   30.486
C   46.945   37.504   29.724
C   47.833   36.341   25.120
C   46.312   31.799   25.811
N   46.288   35.214   29.945
C   45.863   34.369   30.892
C   45.977   34.968   32.275
C   46.030   36.460   31.907
C   46.418   36.443   30.416
C   44.789   37.356   32.166
C   47.220   34.526   33.043
C   46.986   34.398   34.613
C   45.878   35.119   35.366
O   44.795   34.694   35.725
O   46.290   36.415   35.655
N   47.506   36.571   27.501
C   47.540   37.592   28.442
C   48.237   38.749   27.864
C   48.410   38.422   26.480
C   47.912   37.092   26.303
C   48.616   40.053   28.639
C   49.034   39.331   25.459
O   49.153   39.003   24.286
C   49.559   40.745   25.836
N   47.098   34.114   25.752
C   47.499   35.010   24.817
C   47.589   34.336   23.443
C   47.314   32.830   23.763
C   46.922   32.870   25.195
C   46.788   35.063   22.353
C   48.369   31.644   23.553
C   47.780   30.350   23.003
N   46.146   32.781   28.021
C   45.929   31.705   27.176
C   45.310   30.590   27.855
C   45.063   31.126   29.149
C   45.664   32.411   29.227
C   45.142   29.181   27.317
C   44.489   30.903   30.475
O   43.811   29.985   30.812
C   44.554   32.171   31.303
C   44.922   31.717   32.758
O   46.000   31.255   33.101
O   43.672   31.542   33.423
C   43.879   30.884   34.698
C   45.580   37.085   36.759
C   46.717   38.022   37.207
C   46.675   38.970   38.239
C   45.499   38.975   39.229
C   47.819   39.867   38.501
C   47.525   41.364   38.163
C   47.950   42.430   39.296
C   48.678   43.641   38.728
C   49.963   43.886   39.551
C   47.854   44.940   38.781
C   47.024   45.218   40.081
C   47.085   46.655   40.585
C   46.231   47.639   39.721
C   46.874   49.076   39.695
C   44.718   47.589   40.116
C   43.806   47.289   38.972
C   42.287   47.266   39.435
C   41.535   46.175   38.633
C   41.666   46.534   37.082
C   40.170   45.851   39.325
H   46.858   38.434   30.290
H   48.086   36.882   24.205
H   46.056   30.947   25.179
H   45.081   34.763   32.861
H   46.824   36.899   32.511
H   44.680   38.085   31.364
H   44.866   37.851   33.134
H   43.863   36.782   32.210
H   48.064   35.187   32.847
H   47.448   33.505   32.740
H   47.882   34.539   35.217
H   46.846   33.324   34.733
H   47.940   40.896   28.498
H   49.623   40.422   28.444
H   48.523   39.902   29.715
H   50.356   40.794   26.578
H   48.726   41.404   26.084
H   49.974   41.232   24.954
H   48.621   34.444   23.109
H   46.441   32.452   23.230
H   46.210   35.872   22.798
H   46.176   34.409   21.732
H   47.404   35.537   21.589
H   48.865   31.301   24.461
H   49.136   32.080   22.913
H   48.364   30.189   22.097
H   46.701   30.260   22.885
H   47.926   29.573   23.754
H   45.300   28.400   28.060
H   45.884   29.014   26.536
H   44.161   28.953   26.900
H   43.527   32.535   31.315
H   42.898   30.829   35.168
H   44.515   31.505   35.329
H   44.378   29.925   34.835
H   45.265   36.349   37.499
H   44.741   37.682   36.403
H   47.623   37.941   36.608
H   45.863   38.594   40.183
H   44.690   38.358   38.840
H   45.145   40.000   39.338
H   48.737   39.546   38.009
H   47.977   39.714   39.569
H   46.444   41.414   38.030
H   47.949   41.595   37.185
H   48.546   41.862   40.010
H   47.089   42.722   39.897
H   48.890   43.527   37.665
H   49.851   43.403   40.522
H   50.194   44.931   39.759
H   50.845   43.477   39.058
H   47.064   44.775   38.048
H   48.412   45.841   38.526
H   47.297   44.615   40.947
H   45.958   45.009   39.996
H   48.120   46.988   40.508
H   46.852   46.843   41.633
H   46.244   47.174   38.735
H   47.721   49.153   40.377
H   46.165   49.875   39.911
H   47.242   49.028   38.670
H   44.442   48.626   40.309
H   44.499   47.089   41.060
H   44.155   46.370   38.501
H   43.875   48.008   38.156
H   41.749   48.203   39.292
H   42.283   47.038   40.501
H   42.106   45.250   38.702
H   40.717   46.329   36.586
H   42.269   45.758   36.610
H   41.950   47.560   36.847
H   40.040   44.775   39.210
H   39.353   46.351   38.806
H   40.205   46.139   40.376

