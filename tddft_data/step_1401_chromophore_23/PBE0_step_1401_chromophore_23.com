%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1401_chromophore_23 TDDFT with PBE1PBE functional

0 1
Mg   -9.168   41.518   42.478
C   -8.206   38.156   41.618
C   -6.432   42.619   40.668
C   -10.419   44.506   42.720
C   -12.188   39.991   43.508
N   -7.690   40.533   41.219
C   -7.454   39.192   41.038
C   -6.216   38.996   40.126
C   -5.553   40.298   39.949
C   -6.609   41.236   40.635
C   -4.152   40.428   40.649
C   -6.578   38.414   38.721
C   -5.730   37.247   38.291
C   -4.774   37.572   37.176
O   -3.679   38.036   37.355
O   -5.227   37.131   35.947
N   -8.395   43.414   41.907
C   -7.207   43.642   41.295
C   -7.107   45.118   41.189
C   -8.283   45.727   41.667
C   -9.076   44.564   42.176
C   -5.935   45.714   40.557
C   -8.599   47.237   41.631
O   -7.834   47.984   41.097
C   -9.807   47.858   42.393
N   -11.038   42.117   43.140
C   -11.279   43.438   43.149
C   -12.677   43.667   43.597
C   -13.334   42.299   43.792
C   -12.120   41.397   43.415
C   -12.825   44.699   44.748
C   -14.629   41.989   42.920
C   -15.850   41.432   43.668
N   -9.963   39.532   42.795
C   -11.156   39.052   43.238
C   -11.190   37.676   43.315
C   -10.066   37.240   42.596
C   -9.348   38.416   42.350
C   -12.380   36.845   43.691
C   -9.399   36.067   42.138
O   -9.759   34.928   42.188
C   -8.110   36.621   41.445
C   -7.007   36.046   42.119
O   -6.692   36.306   43.266
O   -6.239   35.251   41.255
C   -5.160   34.443   41.981
C   -4.423   37.278   34.700
C   -5.285   37.263   33.435
C   -6.373   37.955   32.941
C   -7.014   39.198   33.578
C   -7.002   37.472   31.673
C   -6.235   37.902   30.383
C   -6.685   36.988   29.192
C   -6.443   37.758   27.864
C   -5.507   37.029   26.946
C   -7.755   37.917   27.134
C   -8.146   39.331   26.835
C   -9.663   39.591   27.010
C   -9.899   40.941   27.816
C   -11.186   40.836   28.496
C   -9.807   42.134   26.734
C   -9.183   43.330   27.428
C   -10.088   44.523   27.428
C   -9.503   45.772   28.296
C   -10.593   46.387   29.172
C   -8.953   46.891   27.345
H   -5.637   43.020   40.035
H   -10.857   45.498   42.845
H   -13.095   39.483   43.844
H   -5.489   38.406   40.686
H   -5.447   40.466   38.877
H   -4.213   40.829   41.660
H   -3.406   41.029   40.129
H   -3.705   39.441   40.762
H   -6.540   39.178   37.944
H   -7.617   38.092   38.793
H   -6.428   36.477   37.962
H   -5.077   36.794   39.037
H   -6.200   46.569   39.935
H   -5.483   45.042   39.829
H   -5.167   45.851   41.318
H   -9.696   47.380   43.366
H   -10.792   47.631   41.985
H   -9.711   48.944   42.396
H   -13.066   44.153   42.702
H   -13.479   42.207   44.869
H   -11.950   45.335   44.882
H   -13.231   44.228   45.644
H   -13.698   45.271   44.436
H   -14.378   41.271   42.139
H   -14.954   42.934   42.485
H   -15.529   41.480   44.708
H   -16.123   40.410   43.406
H   -16.686   42.107   43.484
H   -12.976   37.264   44.501
H   -11.955   35.894   44.014
H   -13.019   36.607   42.840
H   -7.993   36.482   40.371
H   -5.589   33.869   42.802
H   -4.358   35.113   42.292
H   -4.848   33.778   41.176
H   -3.561   36.627   34.558
H   -3.996   38.281   34.686
H   -5.051   36.441   32.759
H   -8.100   39.215   33.490
H   -6.562   40.085   33.132
H   -6.777   39.291   34.637
H   -8.061   37.728   31.706
H   -6.955   36.386   31.584
H   -5.191   37.744   30.653
H   -6.347   38.965   30.172
H   -7.740   36.737   29.300
H   -6.165   36.031   29.163
H   -5.921   38.671   28.152
H   -4.490   37.330   27.199
H   -5.670   37.235   25.888
H   -5.481   35.950   27.100
H   -8.617   37.447   27.607
H   -7.699   37.431   26.159
H   -7.954   39.429   25.767
H   -7.508   40.097   27.277
H   -10.030   38.750   27.598
H   -10.216   39.552   26.072
H   -9.082   41.057   28.528
H   -11.474   41.860   28.732
H   -11.209   40.270   29.427
H   -11.989   40.429   27.881
H   -10.728   42.308   26.177
H   -9.044   41.766   26.047
H   -8.264   43.660   26.944
H   -8.922   43.129   28.468
H   -11.033   44.282   27.914
H   -10.225   44.724   26.365
H   -8.762   45.370   28.988
H   -10.683   45.738   30.043
H   -11.581   46.413   28.713
H   -10.344   47.374   29.562
H   -8.213   47.495   27.869
H   -9.699   47.610   27.009
H   -8.594   46.435   26.422

