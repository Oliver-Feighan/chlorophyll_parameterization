%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1451_chromophore_19 TDDFT with PBE1PBE functional

0 1
Mg   25.600   50.283   26.385
C   23.566   51.560   29.030
C   27.830   49.360   28.749
C   27.650   49.552   23.807
C   23.310   51.697   24.050
N   25.791   50.600   28.609
C   24.689   50.957   29.439
C   25.058   50.559   30.887
C   26.516   50.054   30.782
C   26.719   49.957   29.341
C   27.521   51.040   31.558
C   23.987   49.623   31.556
C   23.397   50.138   32.878
C   24.324   50.832   33.919
O   24.792   51.991   33.777
O   24.382   50.139   35.109
N   27.431   49.491   26.320
C   28.206   49.143   27.419
C   29.368   48.364   26.864
C   29.428   48.535   25.476
C   28.060   49.183   25.170
C   30.339   47.721   27.723
C   30.459   48.065   24.507
O   30.369   48.372   23.290
C   31.757   47.364   24.940
N   25.530   50.639   24.276
C   26.450   50.147   23.451
C   26.183   50.574   22.013
C   24.669   50.995   21.972
C   24.476   51.145   23.527
C   27.215   51.614   21.489
C   23.774   49.889   21.253
C   23.567   48.549   21.952
N   23.805   51.470   26.469
C   22.981   51.945   25.392
C   21.777   52.530   26.040
C   22.016   52.321   27.426
C   23.213   51.743   27.624
C   20.686   53.363   25.344
C   21.448   52.686   28.703
O   20.405   53.307   28.875
C   22.405   52.161   29.819
C   22.901   53.382   30.464
O   23.814   54.041   30.039
O   22.175   53.694   31.609
C   22.519   54.960   32.220
C   25.323   50.581   36.109
C   24.975   49.825   37.325
C   25.180   50.032   38.626
C   25.764   51.408   39.009
C   25.010   48.995   39.736
C   26.106   47.940   39.877
C   25.487   46.502   40.169
C   26.091   45.874   41.404
C   26.845   44.636   40.976
C   25.031   45.599   42.560
C   25.535   45.994   43.929
C   25.268   47.485   44.100
C   26.295   48.104   45.137
C   27.042   47.237   46.179
C   25.653   49.348   45.862
C   26.673   50.493   45.747
C   26.455   51.344   44.500
C   25.991   52.767   44.795
C   24.529   52.981   44.344
C   27.013   53.779   44.171
H   28.630   48.942   29.363
H   28.354   49.326   23.003
H   22.529   51.971   23.337
H   25.081   51.486   31.459
H   26.705   49.132   31.332
H   27.722   51.935   30.970
H   28.517   50.618   31.689
H   27.126   51.496   32.466
H   24.475   48.652   31.642
H   23.108   49.461   30.932
H   22.964   49.245   33.328
H   22.532   50.680   32.496
H   30.687   46.827   27.205
H   29.954   47.411   28.695
H   31.234   48.329   27.852
H   32.197   48.001   25.708
H   32.455   47.396   24.104
H   31.556   46.402   25.411
H   26.334   49.623   21.502
H   24.686   51.947   21.442
H   27.195   52.536   22.070
H   27.116   51.803   20.420
H   28.202   51.160   21.574
H   24.176   49.634   20.272
H   22.861   50.465   21.099
H   22.549   48.460   22.329
H   24.332   48.496   22.727
H   23.676   47.744   21.225
H   20.862   53.400   24.269
H   20.796   54.368   25.752
H   19.756   52.912   25.691
H   22.083   51.410   30.541
H   22.740   55.770   31.525
H   23.346   54.867   32.924
H   21.742   55.428   32.824
H   25.273   51.654   36.295
H   26.356   50.280   35.938
H   24.468   48.883   37.119
H   26.200   51.542   39.999
H   24.962   52.137   38.886
H   26.361   51.823   38.197
H   24.045   48.503   39.609
H   24.853   49.525   40.675
H   26.810   48.192   40.670
H   26.699   47.852   38.967
H   25.686   45.934   39.260
H   24.402   46.498   40.274
H   26.907   46.515   41.736
H   27.065   44.083   41.889
H   27.821   44.991   40.648
H   26.426   43.979   40.213
H   24.903   44.521   42.661
H   24.003   45.903   42.364
H   26.611   45.821   43.917
H   25.103   45.354   44.698
H   24.280   47.535   44.557
H   25.144   48.101   43.210
H   27.091   48.332   44.429
H   26.520   46.305   46.399
H   27.295   47.868   47.031
H   27.962   46.974   45.656
H   25.357   49.237   46.906
H   24.737   49.670   45.367
H   27.719   50.195   45.684
H   26.634   51.029   46.696
H   25.678   50.850   43.916
H   27.379   51.276   43.926
H   25.970   52.961   45.868
H   23.844   53.096   45.184
H   24.238   52.424   43.453
H   24.512   53.937   43.819
H   27.997   53.366   44.393
H   26.851   54.798   44.522
H   27.034   53.642   43.090

