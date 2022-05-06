%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1501_chromophore_23 TDDFT with cam-b3lyp functional

0 1
Mg   -9.653   40.783   41.465
C   -8.621   37.404   40.967
C   -6.805   41.749   39.835
C   -10.738   43.983   41.628
C   -12.556   39.586   42.744
N   -7.954   39.667   40.499
C   -7.768   38.273   40.408
C   -6.423   37.977   39.720
C   -5.708   39.378   39.724
C   -6.883   40.354   40.018
C   -4.542   39.543   40.658
C   -6.742   37.488   38.267
C   -5.923   36.185   37.849
C   -4.959   36.420   36.757
O   -3.919   37.034   36.838
O   -5.344   35.782   35.642
N   -8.968   42.632   40.671
C   -7.640   42.851   40.216
C   -7.348   44.250   40.106
C   -8.584   44.917   40.579
C   -9.514   43.825   40.961
C   -5.981   44.788   39.705
C   -8.683   46.402   40.571
O   -7.779   47.182   40.247
C   -9.941   47.062   41.121
N   -11.503   41.576   41.929
C   -11.675   43.002   42.035
C   -13.090   43.330   42.533
C   -13.717   41.919   42.653
C   -12.521   40.953   42.488
C   -13.121   44.229   43.755
C   -14.958   41.575   41.707
C   -16.196   41.054   42.376
N   -10.414   38.882   41.910
C   -11.624   38.534   42.455
C   -11.683   37.189   42.763
C   -10.498   36.683   42.200
C   -9.803   37.742   41.659
C   -12.825   36.515   43.436
C   -9.695   35.454   42.064
O   -9.823   34.239   42.453
C   -8.507   35.898   41.095
C   -7.206   35.534   41.766
O   -6.926   35.703   42.922
O   -6.456   34.804   40.838
C   -5.263   34.241   41.474
C   -4.513   35.854   34.418
C   -5.373   35.816   33.107
C   -5.766   36.854   32.342
C   -5.503   38.275   32.724
C   -6.539   36.604   31.059
C   -5.835   37.227   29.779
C   -5.858   36.306   28.494
C   -6.373   36.989   27.181
C   -5.640   36.365   25.928
C   -7.903   36.918   27.069
C   -8.569   38.338   27.018
C   -9.935   38.435   27.879
C   -9.775   39.101   29.314
C   -10.359   38.268   30.462
C   -10.453   40.490   29.166
C   -9.512   41.724   29.016
C   -9.242   42.634   30.224
C   -7.698   42.736   30.443
C   -7.179   41.586   31.394
C   -7.261   44.154   30.946
H   -5.836   42.025   39.413
H   -11.035   45.006   41.871
H   -13.560   39.320   43.081
H   -5.739   37.394   40.337
H   -5.375   39.634   38.719
H   -3.689   39.123   40.124
H   -4.724   38.772   41.407
H   -4.393   40.560   41.020
H   -6.590   38.253   37.506
H   -7.750   37.075   38.252
H   -6.634   35.403   37.583
H   -5.337   35.786   38.676
H   -5.545   45.302   40.562
H   -6.141   45.500   38.895
H   -5.382   43.929   39.403
H   -9.991   46.955   42.205
H   -10.904   46.699   40.764
H   -9.835   48.136   40.969
H   -13.507   43.780   41.632
H   -14.011   41.777   43.693
H   -13.609   45.164   43.479
H   -12.128   44.355   44.185
H   -13.741   43.731   44.500
H   -14.730   40.976   40.825
H   -15.311   42.564   41.415
H   -15.852   40.464   43.226
H   -16.814   40.535   41.644
H   -16.856   41.791   42.832
H   -12.473   35.612   43.934
H   -13.540   36.364   42.627
H   -13.285   37.155   44.189
H   -8.610   35.437   40.113
H   -4.701   33.718   40.700
H   -5.415   33.610   42.350
H   -4.621   35.030   41.865
H   -3.755   35.082   34.554
H   -4.075   36.847   34.317
H   -5.449   34.800   32.719
H   -5.493   38.310   33.813
H   -6.438   38.780   32.480
H   -4.629   38.683   32.216
H   -7.438   37.218   31.118
H   -6.869   35.569   30.973
H   -4.784   37.272   30.065
H   -6.084   38.278   29.632
H   -6.518   35.463   28.698
H   -4.850   35.971   28.250
H   -5.856   37.938   27.327
H   -4.903   37.109   25.627
H   -6.319   36.088   25.122
H   -5.148   35.473   26.317
H   -8.339   36.400   27.924
H   -8.121   36.320   26.184
H   -8.889   38.597   26.009
H   -7.993   39.147   27.469
H   -10.305   37.425   28.053
H   -10.683   38.910   27.244
H   -8.724   39.247   29.560
H   -10.123   38.866   31.342
H   -9.757   37.367   30.576
H   -11.438   38.120   30.418
H   -11.179   40.721   29.946
H   -11.179   40.445   28.354
H   -9.736   42.280   28.105
H   -8.568   41.220   28.811
H   -9.700   42.323   31.163
H   -9.624   43.625   29.980
H   -7.180   42.565   29.500
H   -7.945   40.817   31.492
H   -6.830   42.016   32.333
H   -6.329   41.067   30.952
H   -6.792   44.683   30.116
H   -6.540   43.931   31.732
H   -8.045   44.820   31.305
