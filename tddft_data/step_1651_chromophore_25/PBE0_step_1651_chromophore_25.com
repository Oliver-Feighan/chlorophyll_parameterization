%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1651_chromophore_25 TDDFT with PBE1PBE functional

0 1
Mg   -2.220   34.698   26.151
C   -3.584   33.064   29.088
C   -0.677   36.914   28.282
C   -1.489   36.620   23.470
C   -4.378   32.818   24.237
N   -2.253   35.038   28.386
C   -2.694   34.143   29.391
C   -2.236   34.543   30.757
C   -1.394   35.894   30.511
C   -1.432   35.950   28.971
C   -1.945   37.219   31.176
C   -1.335   33.395   31.419
C   -1.952   32.559   32.565
C   -1.121   32.434   33.840
O   -0.181   31.693   34.093
O   -1.486   33.361   34.800
N   -1.008   36.404   25.898
C   -0.371   37.106   26.878
C   0.243   38.245   26.283
C   0.027   38.145   24.836
C   -0.876   37.031   24.665
C   1.115   39.242   27.153
C   0.536   39.096   23.750
O   0.224   38.918   22.562
C   1.423   40.243   24.067
N   -2.953   34.800   24.142
C   -2.443   35.621   23.194
C   -3.338   35.509   21.890
C   -3.944   34.022   22.051
C   -3.826   33.863   23.547
C   -4.379   36.647   21.715
C   -3.212   32.914   21.325
C   -1.816   32.512   21.921
N   -3.624   33.287   26.535
C   -4.306   32.501   25.603
C   -5.016   31.487   26.342
C   -4.743   31.674   27.677
C   -3.926   32.741   27.764
C   -5.856   30.333   25.800
C   -4.957   31.141   28.937
O   -5.566   30.128   29.374
C   -4.214   32.127   29.914
C   -5.255   32.622   30.754
O   -6.177   33.356   30.426
O   -5.079   32.008   31.960
C   -5.976   32.389   33.031
C   -0.588   33.605   35.960
C   -1.279   34.250   37.129
C   -0.664   34.755   38.175
C   0.844   34.961   38.305
C   -1.487   35.184   39.410
C   -1.783   36.680   39.359
C   -1.013   37.530   40.353
C   -1.771   37.740   41.697
C   -1.876   39.133   42.313
C   -1.417   36.627   42.748
C   0.022   36.779   43.240
C   0.022   36.913   44.815
C   1.132   36.018   45.445
C   0.749   35.523   46.859
C   2.523   36.776   45.294
C   3.731   35.882   45.012
C   4.498   36.100   43.676
C   4.228   34.989   42.659
C   5.332   33.893   42.807
C   4.219   35.468   41.228
H   -0.048   37.574   28.884
H   -1.294   37.174   22.549
H   -4.957   32.089   23.665
H   -3.110   34.854   31.329
H   -0.332   35.819   30.746
H   -2.704   37.002   31.927
H   -2.467   37.846   30.454
H   -1.153   37.694   31.755
H   -0.405   33.834   31.779
H   -1.057   32.653   30.671
H   -2.073   31.520   32.258
H   -2.895   32.953   32.945
H   2.096   39.288   26.678
H   1.202   39.104   28.231
H   0.549   40.168   27.051
H   2.402   39.971   24.460
H   0.774   40.930   24.610
H   1.668   40.620   23.074
H   -2.637   35.570   21.057
H   -4.955   33.929   21.655
H   -4.362   36.851   20.645
H   -4.033   37.535   22.245
H   -5.405   36.347   21.929
H   -2.942   33.248   20.323
H   -3.910   32.090   21.177
H   -1.051   32.399   21.154
H   -1.905   31.591   22.498
H   -1.430   33.325   22.535
H   -6.130   30.595   24.778
H   -6.800   30.246   26.337
H   -5.354   29.373   25.921
H   -3.394   31.653   30.455
H   -6.236   31.560   33.689
H   -6.923   32.780   32.657
H   -5.596   33.191   33.664
H   0.248   34.064   35.431
H   -0.148   32.724   36.425
H   -2.369   34.254   37.162
H   1.355   34.863   37.347
H   1.316   34.293   39.027
H   1.143   35.931   38.702
H   -0.843   35.145   40.288
H   -2.439   34.673   39.558
H   -2.840   36.935   39.437
H   -1.495   37.139   38.412
H   -0.824   38.544   40.000
H   -0.042   37.076   40.551
H   -2.802   37.479   41.459
H   -1.285   39.135   43.229
H   -2.884   39.236   42.714
H   -1.537   39.929   41.650
H   -1.538   35.674   42.233
H   -2.124   36.470   43.562
H   0.479   37.722   42.941
H   0.499   35.941   42.732
H   -0.947   36.878   45.313
H   0.308   37.936   45.063
H   1.140   35.143   44.795
H   0.576   34.456   46.715
H   -0.204   35.921   47.207
H   1.476   35.686   47.654
H   2.691   37.394   46.176
H   2.500   37.581   44.560
H   3.331   34.871   45.077
H   4.428   35.966   45.847
H   5.559   36.264   43.862
H   4.233   37.082   43.283
H   3.240   34.550   42.803
H   6.277   34.274   42.421
H   5.010   33.045   42.203
H   5.345   33.762   43.889
H   5.158   35.218   40.733
H   3.882   36.487   41.038
H   3.428   34.848   40.806
