%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_451_chromophore_17 TDDFT with cam-b3lyp functional

0 1
Mg   29.731   58.852   40.326
C   26.674   57.323   39.402
C   31.326   56.216   39.000
C   32.472   60.784   40.682
C   27.785   61.774   41.135
N   29.065   57.039   39.163
C   27.795   56.542   39.046
C   27.735   55.182   38.381
C   29.263   54.732   38.527
C   29.959   56.026   39.018
C   29.439   53.486   39.512
C   27.336   55.300   36.921
C   26.161   54.407   36.426
C   26.361   53.698   35.140
O   26.271   52.540   34.922
O   26.617   54.611   34.153
N   31.740   58.558   39.987
C   32.156   57.416   39.368
C   33.608   57.510   39.294
C   34.018   58.821   39.861
C   32.720   59.457   40.171
C   34.518   56.385   38.867
C   35.449   59.425   40.059
O   36.417   58.659   39.905
C   35.674   60.809   40.660
N   30.020   60.990   40.930
C   31.263   61.434   40.930
C   31.218   62.965   41.302
C   29.722   63.279   41.369
C   29.137   61.925   41.202
C   32.014   63.447   42.557
C   29.161   64.393   40.373
C   28.044   65.240   40.912
N   27.632   59.403   40.404
C   27.052   60.610   40.812
C   25.670   60.442   40.780
C   25.493   59.171   40.242
C   26.690   58.627   39.926
C   24.612   61.523   41.024
C   24.462   58.174   39.889
O   23.242   58.324   39.995
C   25.166   56.974   39.218
C   24.830   55.744   39.950
O   25.541   55.201   40.805
O   23.591   55.328   39.612
C   22.963   54.139   40.232
C   27.169   53.924   32.942
C   27.683   54.984   32.023
C   28.555   54.801   30.997
C   29.016   53.406   30.332
C   29.128   56.072   30.288
C   30.523   55.937   29.688
C   31.417   57.169   29.949
C   32.733   57.154   29.101
C   33.006   58.521   28.443
C   33.982   56.753   29.974
C   35.184   56.417   29.031
C   35.865   55.010   29.165
C   37.270   55.010   29.741
C   37.936   53.587   29.574
C   37.117   55.416   31.245
C   37.771   56.806   31.638
C   38.279   56.828   33.080
C   39.803   57.364   33.138
C   40.470   56.789   34.485
C   39.836   58.929   33.080
H   31.975   55.457   38.559
H   33.355   61.426   40.668
H   27.131   62.595   41.435
H   27.092   54.485   38.918
H   29.703   54.407   37.584
H   28.709   52.691   39.359
H   29.175   53.834   40.511
H   30.445   53.066   39.499
H   28.138   55.171   36.194
H   27.040   56.330   36.721
H   25.299   55.053   36.261
H   26.003   53.683   37.226
H   34.118   55.844   38.010
H   34.548   55.612   39.635
H   35.560   56.608   38.637
H   35.180   61.693   40.256
H   36.729   61.028   40.500
H   35.495   60.836   41.735
H   31.591   63.466   40.408
H   29.585   63.703   42.364
H   32.759   62.718   42.875
H   31.369   63.483   43.434
H   32.496   64.385   42.282
H   28.683   63.920   39.515
H   29.972   65.054   40.068
H   27.102   64.909   40.474
H   28.200   66.280   40.627
H   27.930   65.107   41.988
H   24.681   61.981   42.011
H   23.679   61.034   40.744
H   24.908   62.274   40.291
H   24.911   56.796   38.174
H   23.648   53.498   40.788
H   22.421   53.471   39.562
H   22.322   54.578   40.996
H   26.340   53.554   32.338
H   28.005   53.245   33.107
H   27.511   56.026   32.294
H   28.983   53.614   29.262
H   28.243   52.640   30.400
H   29.933   53.090   30.829
H   29.292   56.903   30.975
H   28.423   56.310   29.492
H   30.345   55.899   28.613
H   31.099   55.038   29.906
H   31.629   57.352   31.002
H   30.798   58.049   29.775
H   32.750   56.410   28.305
H   34.079   58.706   28.408
H   32.513   59.287   29.042
H   32.540   58.611   27.461
H   33.754   55.816   30.484
H   34.236   57.609   30.599
H   35.949   57.160   29.253
H   34.905   56.446   27.978
H   35.702   54.383   28.288
H   35.314   54.390   29.873
H   37.784   55.870   29.313
H   38.980   53.780   29.331
H   37.436   53.182   28.694
H   37.768   52.954   30.446
H   37.505   54.676   31.945
H   36.085   55.615   31.536
H   37.040   57.587   31.428
H   38.602   57.032   30.969
H   38.206   55.835   33.525
H   37.725   57.573   33.650
H   40.352   56.833   32.361
H   39.810   55.993   34.830
H   40.607   57.506   35.294
H   41.481   56.453   34.256
H   38.996   59.522   33.443
H   40.119   59.029   32.033
H   40.587   59.348   33.749

