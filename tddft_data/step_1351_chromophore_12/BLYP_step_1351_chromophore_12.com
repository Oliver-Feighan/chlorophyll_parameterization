%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1351_chromophore_12 TDDFT with blyp functional

0 1
Mg   47.136   15.795   28.866
C   45.102   15.459   31.584
C   49.135   18.050   30.829
C   49.117   16.272   26.131
C   44.819   14.220   26.847
N   46.998   16.799   30.904
C   46.164   16.394   31.897
C   46.569   16.941   33.271
C   47.480   18.102   32.841
C   47.863   17.695   31.414
C   46.824   19.554   32.915
C   47.218   15.845   34.158
C   46.933   16.010   35.696
C   48.045   15.816   36.713
O   48.979   15.040   36.587
O   47.846   16.800   37.698
N   48.989   16.898   28.576
C   49.626   17.694   29.523
C   50.924   18.032   28.896
C   50.994   17.583   27.517
C   49.667   16.871   27.316
C   51.969   19.003   29.623
C   52.104   17.845   26.527
O   52.094   17.403   25.424
C   53.313   18.660   26.899
N   46.952   15.468   26.784
C   47.882   15.624   25.898
C   47.608   14.931   24.588
C   46.185   14.319   24.783
C   45.949   14.699   26.279
C   47.788   15.841   23.307
C   46.132   12.805   24.398
C   46.690   11.753   25.402
N   45.298   14.939   29.121
C   44.447   14.385   28.208
C   43.366   13.795   28.955
C   43.566   14.242   30.343
C   44.752   14.917   30.334
C   42.206   13.083   28.320
C   43.076   14.130   31.699
O   42.065   13.554   32.132
C   44.095   14.943   32.566
C   43.357   15.977   33.279
O   42.872   16.993   32.851
O   43.486   15.714   34.623
C   43.309   16.923   35.419
C   48.828   16.743   38.834
C   49.154   18.105   39.489
C   50.319   18.742   39.655
C   51.690   18.139   39.334
C   50.235   20.194   40.224
C   50.710   21.295   39.376
C   51.381   22.478   40.014
C   52.936   22.538   39.768
C   53.694   22.268   41.081
C   53.357   23.856   39.046
C   53.507   23.637   37.524
C   52.490   24.339   36.564
C   52.894   24.183   35.057
C   54.239   24.714   34.689
C   51.750   24.525   34.178
C   51.678   23.639   32.949
C   50.607   24.207   31.982
C   50.215   23.264   30.786
C   50.178   23.974   29.465
C   48.917   22.472   31.155
H   49.770   18.647   31.487
H   49.717   16.292   25.219
H   44.152   13.555   26.295
H   45.633   17.274   33.719
H   48.424   17.973   33.369
H   46.928   19.969   31.913
H   47.374   20.204   33.595
H   45.787   19.456   33.236
H   48.291   15.746   33.995
H   46.758   14.881   33.941
H   46.130   15.343   36.009
H   46.527   17.016   35.810
H   52.985   18.619   29.713
H   51.686   19.378   30.607
H   52.036   19.930   29.053
H   53.998   18.653   26.051
H   53.903   18.211   27.698
H   53.015   19.651   27.242
H   48.337   14.124   24.515
H   45.591   14.830   24.025
H   48.845   15.820   23.041
H   47.519   16.890   23.433
H   47.165   15.411   22.522
H   46.614   12.609   23.440
H   45.080   12.561   24.248
H   47.661   11.394   25.060
H   45.968   10.937   25.400
H   46.852   12.195   26.385
H   42.493   12.158   27.819
H   41.821   13.679   27.494
H   41.315   12.984   28.940
H   44.544   14.266   33.293
H   43.800   17.715   34.854
H   43.978   16.639   36.232
H   42.284   17.157   35.705
H   49.794   16.369   38.495
H   48.394   16.086   39.587
H   48.200   18.600   39.671
H   52.482   18.877   39.459
H   51.686   17.633   38.368
H   51.979   17.361   40.040
H   50.924   20.245   41.067
H   49.252   20.421   40.637
H   49.837   21.691   38.856
H   51.229   20.818   38.545
H   51.287   22.490   41.099
H   50.899   23.363   39.599
H   53.153   21.674   39.140
H   54.312   23.150   41.244
H   54.263   21.339   41.034
H   53.044   22.216   41.954
H   54.315   24.248   39.387
H   52.657   24.673   39.222
H   53.400   22.587   37.252
H   54.513   23.898   37.194
H   52.337   25.384   36.837
H   51.573   23.760   36.672
H   53.027   23.107   34.949
H   54.283   25.103   33.672
H   54.967   23.923   34.868
H   54.641   25.570   35.232
H   51.861   25.534   33.780
H   50.801   24.413   34.703
H   51.586   22.571   33.145
H   52.653   23.692   32.463
H   50.780   25.226   31.637
H   49.740   24.238   32.642
H   50.972   22.492   30.647
H   49.404   23.671   28.761
H   51.148   23.845   28.985
H   50.164   25.061   29.547
H   49.134   22.087   32.151
H   48.739   21.607   30.516
H   48.058   23.140   31.089
