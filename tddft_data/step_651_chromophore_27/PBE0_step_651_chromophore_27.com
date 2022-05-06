%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_651_chromophore_27 TDDFT with PBE1PBE functional

0 1
Mg   -5.245   25.333   26.922
C   -3.737   27.121   29.570
C   -6.131   22.988   29.375
C   -6.485   23.536   24.521
C   -3.748   27.563   24.729
N   -5.009   25.138   29.275
C   -4.440   26.036   30.098
C   -4.845   25.762   31.538
C   -5.388   24.316   31.438
C   -5.497   24.113   29.970
C   -4.604   23.017   32.060
C   -5.893   26.785   32.106
C   -5.735   27.202   33.589
C   -6.991   27.047   34.445
O   -8.069   27.560   34.200
O   -6.752   26.381   35.582
N   -6.298   23.616   26.983
C   -6.601   22.801   28.059
C   -7.494   21.731   27.672
C   -7.726   21.858   26.271
C   -6.845   23.072   25.855
C   -8.119   20.792   28.643
C   -8.576   21.039   25.340
O   -8.636   21.257   24.131
C   -9.376   19.917   25.906
N   -4.901   25.436   24.850
C   -5.742   24.630   24.035
C   -5.689   25.113   22.557
C   -4.777   26.410   22.653
C   -4.472   26.530   24.182
C   -5.121   24.088   21.588
C   -5.562   27.645   22.107
C   -5.517   27.612   20.629
N   -3.997   26.953   27.054
C   -3.464   27.781   26.080
C   -2.660   28.844   26.692
C   -2.796   28.589   28.090
C   -3.497   27.447   28.223
C   -1.979   30.003   26.050
C   -2.420   29.128   29.381
O   -1.721   30.123   29.533
C   -3.181   28.242   30.418
C   -2.215   27.847   31.506
O   -1.185   27.216   31.383
O   -2.766   28.209   32.747
C   -1.771   27.853   33.799
C   -7.855   26.345   36.617
C   -8.746   25.076   36.395
C   -8.660   23.805   36.958
C   -7.546   23.517   37.935
C   -9.673   22.694   36.584
C   -11.138   23.011   37.005
C   -11.632   21.990   38.099
C   -12.713   21.043   37.552
C   -14.101   21.607   37.443
C   -12.833   19.706   38.400
C   -12.302   18.457   37.682
C   -11.325   17.615   38.563
C   -9.865   17.647   38.096
C   -9.246   19.098   38.355
C   -8.908   16.562   38.783
C   -7.861   15.962   37.860
C   -6.382   16.339   38.349
C   -5.309   15.248   38.026
C   -4.617   14.669   39.261
C   -4.159   15.648   37.127
H   -6.512   22.231   30.063
H   -7.008   22.907   23.797
H   -3.389   28.337   24.047
H   -4.020   25.697   32.248
H   -6.401   24.283   31.838
H   -3.723   23.149   32.688
H   -4.311   22.355   31.244
H   -5.282   22.407   32.657
H   -6.895   26.489   31.795
H   -5.806   27.666   31.470
H   -5.370   28.229   33.593
H   -4.938   26.612   34.040
H   -9.154   20.982   28.359
H   -7.991   20.863   29.723
H   -7.902   19.799   28.248
H   -9.973   20.313   26.727
H   -8.703   19.121   26.225
H   -10.082   19.553   25.159
H   -6.625   25.484   22.141
H   -3.861   26.212   22.095
H   -5.503   24.148   20.569
H   -5.156   23.085   22.013
H   -4.037   24.194   21.538
H   -5.083   28.552   22.475
H   -6.619   27.582   22.362
H   -4.554   27.279   20.241
H   -5.699   28.601   20.209
H   -6.334   27.140   20.083
H   -2.502   30.954   26.160
H   -1.815   29.983   24.973
H   -1.015   30.074   26.554
H   -3.952   28.864   30.872
H   -0.726   28.048   33.558
H   -1.845   26.798   34.061
H   -1.842   28.515   34.662
H   -8.471   27.244   36.628
H   -7.411   26.194   37.601
H   -9.641   25.278   35.807
H   -7.890   23.595   38.967
H   -6.704   24.202   37.834
H   -7.303   22.469   37.763
H   -9.321   21.743   36.983
H   -9.703   22.459   35.520
H   -11.825   22.886   36.168
H   -11.185   23.989   37.484
H   -12.051   22.559   38.930
H   -10.809   21.500   38.620
H   -12.429   20.739   36.544
H   -14.941   20.924   37.564
H   -14.101   22.053   36.448
H   -14.211   22.488   38.075
H   -13.896   19.519   38.551
H   -12.391   19.810   39.391
H   -11.610   18.788   36.909
H   -13.063   17.853   37.188
H   -11.691   16.591   38.494
H   -11.268   17.856   39.625
H   -9.814   17.378   37.041
H   -9.298   19.674   37.431
H   -8.179   18.914   38.481
H   -9.870   19.675   39.038
H   -9.513   15.753   39.193
H   -8.507   17.007   39.693
H   -7.963   16.388   36.862
H   -8.011   14.882   37.883
H   -6.282   16.578   39.408
H   -6.218   17.287   37.837
H   -5.831   14.405   37.574
H   -4.111   13.794   38.853
H   -5.266   14.415   40.100
H   -3.862   15.350   39.654
H   -3.619   16.449   37.632
H   -4.584   16.089   36.225
H   -3.405   14.920   36.828

