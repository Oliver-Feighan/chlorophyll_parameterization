%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_551_chromophore_11 TDDFT with PBE1PBE functional

0 1
Mg   52.080   23.863   44.535
C   49.138   25.838   43.953
C   50.023   21.126   44.390
C   54.802   22.040   44.224
C   53.867   26.731   44.734
N   49.766   23.539   44.129
C   48.822   24.469   44.006
C   47.395   23.825   43.854
C   47.680   22.301   44.240
C   49.240   22.271   44.288
C   47.088   21.729   45.496
C   46.621   24.045   42.558
C   47.352   23.643   41.264
C   47.218   24.483   39.943
O   46.797   25.592   39.813
O   47.808   23.662   38.952
N   52.351   21.825   44.353
C   51.440   20.852   44.335
C   52.063   19.596   44.134
C   53.443   19.765   44.216
C   53.590   21.209   44.324
C   51.277   18.342   44.014
C   54.480   18.673   44.097
O   54.204   17.433   43.948
C   55.981   18.898   44.124
N   53.977   24.312   44.272
C   55.003   23.439   44.289
C   56.351   24.041   44.179
C   56.055   25.565   44.377
C   54.543   25.555   44.438
C   57.412   23.372   45.125
C   56.668   26.562   43.359
C   57.828   27.467   43.811
N   51.644   25.954   44.484
C   52.459   26.972   44.559
C   51.751   28.204   44.671
C   50.375   27.774   44.400
C   50.421   26.409   44.282
C   52.310   29.543   44.771
C   48.944   28.230   44.345
O   48.414   29.337   44.531
C   48.153   27.035   43.840
C   47.888   27.408   42.439
O   48.634   27.266   41.492
O   46.768   28.172   42.427
C   46.472   28.688   41.102
C   47.688   24.167   37.597
C   48.300   23.032   36.592
C   47.941   21.712   36.498
C   46.898   21.049   37.276
C   48.663   20.746   35.645
C   50.052   20.495   36.002
C   51.025   20.368   34.849
C   51.936   19.149   35.013
C   53.058   19.404   36.005
C   52.542   18.612   33.694
C   51.646   17.812   32.816
C   52.075   16.338   32.938
C   51.145   15.493   32.128
C   50.001   14.788   32.885
C   51.877   14.465   31.259
C   52.642   15.096   29.962
C   52.404   14.337   28.609
C   53.652   13.479   28.171
C   54.744   14.477   27.807
C   53.150   12.740   26.918
H   49.470   20.193   44.520
H   55.739   21.519   44.017
H   54.412   27.675   44.797
H   46.783   24.239   44.655
H   47.511   21.628   43.400
H   46.687   20.743   45.259
H   46.210   22.306   45.786
H   47.790   21.704   46.330
H   46.251   25.070   42.579
H   45.693   23.474   42.538
H   46.946   22.645   41.104
H   48.424   23.645   41.465
H   50.213   18.491   43.831
H   51.372   17.801   44.956
H   51.630   17.762   43.161
H   56.251   19.350   43.169
H   56.528   17.992   44.385
H   56.137   19.619   44.926
H   56.809   23.822   43.214
H   56.280   25.694   45.436
H   56.907   22.578   45.675
H   57.708   24.138   45.842
H   58.272   23.020   44.555
H   55.934   27.151   42.810
H   57.021   25.826   42.637
H   57.702   27.513   44.893
H   57.720   28.475   43.411
H   58.786   27.096   43.446
H   53.340   29.687   44.444
H   52.287   29.847   45.817
H   51.747   30.210   44.119
H   47.272   26.812   44.442
H   45.402   28.862   40.983
H   46.816   28.120   40.238
H   46.808   29.717   40.969
H   48.466   24.925   37.505
H   46.690   24.546   37.378
H   49.182   23.428   36.089
H   47.184   20.445   38.137
H   46.336   20.341   36.668
H   46.120   21.746   37.588
H   48.637   21.111   34.619
H   48.159   19.780   35.633
H   50.195   19.666   36.694
H   50.447   21.290   36.635
H   51.573   21.301   34.712
H   50.463   20.286   33.919
H   51.434   18.278   35.433
H   52.921   20.248   36.681
H   54.089   19.393   35.652
H   53.065   18.566   36.702
H   53.477   18.067   33.828
H   52.864   19.447   33.071
H   51.694   18.231   31.811
H   50.599   17.894   33.106
H   51.953   16.142   34.003
H   53.148   16.203   32.805
H   50.556   16.140   31.479
H   50.424   14.027   33.540
H   49.353   14.336   32.135
H   49.392   15.380   33.568
H   51.189   13.687   30.928
H   52.501   13.984   32.012
H   53.691   15.318   30.159
H   52.419   16.144   29.764
H   52.149   15.058   27.832
H   51.587   13.615   28.626
H   53.860   12.815   29.010
H   55.244   14.101   26.915
H   55.549   14.486   28.542
H   54.435   15.512   27.664
H   53.073   11.731   27.323
H   53.941   12.653   26.173
H   52.123   12.921   26.600

