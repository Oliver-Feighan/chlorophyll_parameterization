%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1201_chromophore_16 TDDFT with PBE1PBE functional

0 1
Mg   41.163   41.760   26.924
C   40.381   44.018   29.670
C   41.832   39.410   29.345
C   41.840   39.825   24.516
C   40.410   44.426   24.813
N   41.272   41.837   29.321
C   40.864   42.822   30.184
C   40.930   42.330   31.663
C   41.585   40.890   31.464
C   41.569   40.678   29.963
C   43.002   40.611   32.080
C   39.643   42.281   32.493
C   39.841   42.321   33.985
C   38.834   42.997   34.859
O   37.718   43.405   34.553
O   39.353   43.112   36.099
N   41.781   39.834   26.912
C   41.983   39.019   27.967
C   42.384   37.682   27.420
C   42.351   37.774   26.028
C   42.008   39.169   25.762
C   42.728   36.506   28.336
C   42.627   36.745   24.946
O   42.687   36.904   23.731
C   42.774   35.288   25.381
N   41.412   42.192   24.840
C   41.559   41.139   24.053
C   41.389   41.589   22.624
C   40.723   43.022   22.750
C   40.827   43.236   24.201
C   42.672   41.366   21.723
C   39.273   43.249   22.199
C   38.310   42.047   22.554
N   40.774   43.887   27.137
C   40.434   44.812   26.163
C   40.167   46.037   26.776
C   39.968   45.680   28.131
C   40.440   44.393   28.313
C   39.934   47.294   26.046
C   39.466   46.243   29.390
O   38.929   47.283   29.677
C   39.657   45.118   30.454
C   40.563   45.717   31.444
O   41.741   46.053   31.258
O   39.840   45.853   32.632
C   40.644   46.401   33.761
C   38.552   43.860   37.133
C   38.991   43.395   38.519
C   38.656   43.946   39.703
C   37.731   45.149   39.906
C   39.205   43.320   40.990
C   38.332   42.119   41.470
C   38.929   40.726   41.129
C   38.894   39.753   42.294
C   40.364   39.510   42.662
C   37.976   38.449   42.219
C   36.760   38.490   43.219
C   37.113   38.122   44.667
C   36.228   38.721   45.775
C   37.108   39.840   46.497
C   35.760   37.578   46.741
C   34.676   36.575   46.122
C   33.507   36.491   47.117
C   32.207   35.927   46.403
C   31.216   35.391   47.442
C   31.718   37.138   45.502
H   41.966   38.603   30.068
H   41.810   39.132   23.673
H   40.028   45.199   24.143
H   41.643   42.951   32.205
H   40.976   40.113   31.926
H   43.353   41.570   32.461
H   43.792   40.256   31.419
H   42.907   39.886   32.888
H   39.215   41.280   32.446
H   38.819   42.938   32.214
H   40.822   42.769   34.148
H   40.002   41.303   34.341
H   42.018   35.686   28.230
H   42.779   36.776   29.391
H   43.684   36.100   28.007
H   42.886   34.644   24.509
H   41.847   35.007   25.881
H   43.649   35.324   26.031
H   40.641   40.961   22.140
H   41.377   43.804   22.366
H   43.551   41.502   22.353
H   42.437   42.109   20.960
H   42.618   40.348   21.337
H   39.258   43.194   21.111
H   38.740   44.060   22.695
H   38.550   41.251   21.849
H   37.271   42.349   22.423
H   38.497   41.709   23.573
H   40.406   47.394   25.068
H   40.254   48.076   26.735
H   38.852   47.299   25.913
H   38.738   44.730   30.894
H   41.392   45.705   34.141
H   39.953   46.782   34.513
H   41.158   47.277   33.365
H   37.481   43.713   36.998
H   38.801   44.911   36.985
H   39.691   42.563   38.599
H   37.013   44.920   40.693
H   37.195   45.481   39.016
H   38.385   45.944   40.263
H   39.293   44.054   41.791
H   40.240   43.025   40.819
H   37.352   42.146   40.994
H   38.181   42.287   42.537
H   39.916   40.887   40.696
H   38.266   40.386   40.334
H   38.511   40.299   43.156
H   41.026   39.275   41.828
H   40.562   38.709   43.375
H   40.712   40.410   43.169
H   38.694   37.637   42.334
H   37.572   38.555   41.212
H   36.065   37.759   42.804
H   36.263   39.457   43.299
H   38.152   38.401   44.840
H   37.118   37.034   44.737
H   35.336   39.233   45.417
H   36.379   40.625   46.697
H   37.968   40.252   45.970
H   37.530   39.436   47.417
H   35.417   38.113   47.626
H   36.602   37.031   47.163
H   35.211   35.625   46.121
H   34.397   36.975   45.147
H   33.268   37.446   47.586
H   33.688   35.638   47.771
H   32.456   35.051   45.804
H   30.348   35.127   46.838
H   30.791   36.158   48.090
H   31.571   34.619   48.124
H   30.646   37.318   45.581
H   31.909   36.819   44.477
H   32.300   38.045   45.666

