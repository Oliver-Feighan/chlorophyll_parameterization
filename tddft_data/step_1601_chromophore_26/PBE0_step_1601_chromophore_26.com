%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1601_chromophore_26 TDDFT with PBE1PBE functional

0 1
Mg   -9.929   17.555   42.137
C   -6.465   16.946   42.809
C   -9.188   20.747   41.439
C   -13.181   18.073   41.435
C   -10.511   14.065   42.565
N   -7.931   18.661   42.064
C   -6.740   18.264   42.389
C   -5.737   19.394   42.146
C   -6.694   20.662   42.089
C   -8.024   20.005   41.787
C   -6.745   21.500   43.409
C   -4.817   19.126   40.911
C   -5.376   18.530   39.649
C   -5.100   19.333   38.359
O   -4.184   20.082   38.034
O   -5.883   18.908   37.342
N   -11.034   19.213   41.570
C   -10.541   20.413   41.391
C   -11.565   21.399   41.211
C   -12.776   20.626   41.097
C   -12.350   19.242   41.343
C   -11.423   22.880   40.933
C   -14.174   21.193   40.920
O   -14.365   22.402   40.995
C   -15.414   20.510   40.543
N   -11.643   16.236   41.956
C   -12.925   16.673   41.732
C   -13.904   15.551   41.887
C   -13.052   14.265   42.173
C   -11.652   14.854   42.273
C   -14.879   15.681   42.964
C   -13.170   13.242   41.023
C   -14.139   11.971   41.250
N   -8.805   15.816   42.447
C   -9.144   14.528   42.583
C   -7.947   13.675   42.818
C   -6.893   14.641   42.943
C   -7.444   15.887   42.765
C   -7.830   12.227   42.948
C   -5.438   14.783   43.262
O   -4.557   13.916   43.329
C   -5.192   16.278   43.209
C   -4.614   16.746   44.474
O   -3.597   17.407   44.534
O   -5.450   16.540   45.556
C   -5.201   17.270   46.825
C   -5.486   19.215   35.973
C   -6.710   18.674   35.257
C   -7.405   19.334   34.260
C   -7.096   20.693   33.667
C   -8.254   18.437   33.396
C   -9.651   18.066   33.892
C   -10.641   17.788   32.788
C   -11.655   18.899   32.482
C   -13.038   18.241   32.212
C   -11.241   19.880   31.333
C   -11.682   21.367   31.713
C   -10.620   22.385   31.315
C   -11.152   23.419   30.279
C   -10.158   24.583   29.999
C   -11.561   22.638   28.968
C   -12.564   23.498   28.131
C   -12.121   23.905   26.705
C   -13.096   23.362   25.571
C   -12.366   22.336   24.657
C   -13.613   24.486   24.705
H   -9.013   21.796   41.192
H   -14.236   18.259   41.223
H   -10.673   12.990   42.663
H   -5.099   19.389   43.031
H   -6.297   21.246   41.259
H   -7.771   21.438   43.773
H   -6.534   22.499   43.028
H   -5.881   21.100   43.939
H   -4.286   18.214   41.181
H   -4.067   19.915   40.850
H   -6.446   18.506   39.852
H   -4.986   17.541   39.404
H   -11.368   23.104   39.867
H   -10.454   23.216   41.302
H   -12.181   23.445   41.476
H   -15.549   19.865   41.411
H   -15.290   19.963   39.608
H   -16.238   21.222   40.484
H   -14.382   15.511   40.908
H   -13.181   13.914   43.197
H   -15.686   16.358   42.686
H   -14.388   16.128   43.829
H   -15.241   14.710   43.302
H   -12.161   12.865   40.858
H   -13.498   13.841   40.174
H   -15.035   11.976   40.630
H   -14.446   12.124   42.285
H   -13.636   11.010   41.141
H   -8.638   11.849   42.320
H   -7.993   11.993   44.000
H   -6.789   12.055   42.677
H   -4.308   16.437   42.592
H   -5.094   18.345   46.680
H   -4.304   16.878   47.304
H   -6.100   17.114   47.422
H   -4.675   18.554   35.670
H   -5.247   20.270   35.836
H   -7.187   17.723   35.495
H   -6.611   20.581   32.697
H   -6.380   21.292   34.229
H   -7.994   21.273   33.455
H   -7.711   17.498   33.288
H   -8.421   18.774   32.373
H   -10.112   18.850   34.493
H   -9.551   17.198   34.544
H   -11.269   16.984   33.171
H   -10.155   17.399   31.893
H   -11.784   19.471   33.401
H   -13.471   17.931   33.162
H   -12.892   17.366   31.579
H   -13.731   18.976   31.803
H   -11.629   19.596   30.355
H   -10.153   19.844   31.279
H   -11.744   21.463   32.797
H   -12.653   21.611   31.283
H   -9.812   21.910   30.758
H   -10.263   22.860   32.229
H   -12.101   23.853   30.591
H   -10.567   25.315   29.302
H   -9.199   24.174   29.681
H   -10.037   25.050   30.976
H   -12.048   21.670   29.092
H   -10.677   22.437   28.364
H   -12.758   24.383   28.736
H   -13.526   22.987   28.194
H   -11.120   23.573   26.428
H   -12.080   24.993   26.675
H   -13.820   22.748   26.106
H   -13.027   21.982   23.865
H   -12.057   21.488   25.268
H   -11.475   22.799   24.232
H   -13.045   24.538   23.776
H   -13.488   25.459   25.180
H   -14.694   24.432   24.581

