%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_251_chromophore_23 ZINDO

0 1
Mg   -9.427   40.512   41.960
C   -8.276   37.271   41.013
C   -7.189   41.707   39.821
C   -11.063   43.469   41.994
C   -12.243   38.824   43.346
N   -7.886   39.632   40.718
C   -7.508   38.285   40.529
C   -6.306   38.117   39.547
C   -5.961   39.601   39.236
C   -7.016   40.390   40.040
C   -4.487   39.821   39.615
C   -6.546   37.266   38.226
C   -5.773   35.958   38.103
C   -4.886   35.772   36.855
O   -3.650   35.837   36.890
O   -5.596   35.629   35.694
N   -9.143   42.334   41.144
C   -8.118   42.628   40.245
C   -8.094   44.005   39.957
C   -9.107   44.611   40.677
C   -9.835   43.523   41.298
C   -7.101   44.802   39.032
C   -9.331   46.105   40.686
O   -8.596   46.898   40.177
C   -10.578   46.611   41.565
N   -11.401   41.074   42.511
C   -11.806   42.380   42.462
C   -13.340   42.402   42.854
C   -13.696   41.008   43.397
C   -12.372   40.224   43.055
C   -13.759   43.567   43.705
C   -14.849   40.427   42.588
C   -15.681   39.356   43.242
N   -10.030   38.457   42.379
C   -11.169   37.955   42.962
C   -11.088   36.543   43.061
C   -9.916   36.156   42.369
C   -9.388   37.383   41.877
C   -12.099   35.620   43.613
C   -9.038   35.083   41.883
O   -9.114   33.866   42.099
C   -7.913   35.746   40.999
C   -6.650   35.630   41.678
O   -6.376   36.353   42.665
O   -5.964   34.635   41.082
C   -4.539   34.427   41.459
C   -4.742   35.378   34.470
C   -5.642   35.359   33.237
C   -6.023   36.437   32.543
C   -5.617   37.915   32.812
C   -6.861   36.252   31.276
C   -6.042   36.309   29.936
C   -6.726   37.386   28.995
C   -5.823   38.643   28.601
C   -4.891   38.275   27.343
C   -6.506   39.994   28.262
C   -6.251   41.043   29.329
C   -7.477   41.827   29.860
C   -7.079   43.366   29.961
C   -7.256   43.879   31.458
C   -7.887   44.211   28.976
C   -7.016   44.985   27.966
C   -6.542   46.328   28.472
C   -5.037   46.289   28.869
C   -4.896   47.352   30.043
C   -4.081   46.746   27.692
H   -6.439   42.138   39.154
H   -11.530   44.455   42.043
H   -13.019   38.259   43.866
H   -5.467   37.657   40.069
H   -6.103   39.871   38.190
H   -4.303   40.893   39.550
H   -3.753   39.315   38.988
H   -4.268   39.500   40.634
H   -6.117   37.885   37.438
H   -7.614   37.121   38.063
H   -6.500   35.147   38.158
H   -5.087   35.868   38.946
H   -7.731   45.355   38.335
H   -6.441   44.138   38.474
H   -6.377   45.435   39.545
H   -10.520   47.699   41.555
H   -10.393   46.151   42.536
H   -11.421   46.180   41.025
H   -13.837   42.546   41.894
H   -13.958   40.943   44.453
H   -14.599   43.253   44.324
H   -14.039   44.358   43.009
H   -12.953   44.015   44.285
H   -14.538   40.038   41.618
H   -15.546   41.196   42.255
H   -15.479   39.253   44.308
H   -15.452   38.411   42.749
H   -16.736   39.486   43.001
H   -12.698   36.096   44.390
H   -11.534   34.784   44.026
H   -12.748   35.368   42.774
H   -7.890   35.411   39.962
H   -4.473   33.343   41.355
H   -4.284   34.796   42.452
H   -3.864   34.765   40.672
H   -4.225   34.423   34.566
H   -3.980   36.146   34.340
H   -5.790   34.394   32.753
H   -5.507   37.980   33.895
H   -6.421   38.592   32.523
H   -4.709   38.151   32.258
H   -7.687   36.961   31.225
H   -7.342   35.284   31.417
H   -6.045   35.351   29.416
H   -5.005   36.623   30.049
H   -7.724   37.767   29.212
H   -6.796   36.757   28.108
H   -5.096   38.831   29.391
H   -4.432   37.294   27.463
H   -4.034   38.949   27.349
H   -5.380   38.308   26.369
H   -7.573   39.790   28.176
H   -6.094   40.254   27.287
H   -5.692   41.657   28.623
H   -5.663   40.766   30.204
H   -7.808   41.320   30.767
H   -8.359   41.702   29.232
H   -5.997   43.358   29.827
H   -7.976   43.252   31.983
H   -7.724   44.859   31.557
H   -6.286   43.834   31.953
H   -8.528   44.975   29.417
H   -8.727   43.754   28.453
H   -7.716   45.070   27.135
H   -6.101   44.462   27.689
H   -7.173   46.756   29.250
H   -6.743   46.967   27.613
H   -4.672   45.302   29.155
H   -3.984   47.055   30.562
H   -5.686   47.266   30.788
H   -4.793   48.368   29.662
H   -3.311   45.975   27.660
H   -3.643   47.728   27.870
H   -4.580   46.810   26.725

