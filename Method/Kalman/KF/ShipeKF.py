import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
# 设置黑体字符
plt.rcParams["font.sans-serif"]=["SimHei"]

# 设定状态参数
# 单位时间
dt = 1 
# x初始位置（X位置，Y位置，X加速度，Y加速度）
x = np.array([[0.0],[0.0],[15.0],[5.0]]) 
# 状态转移矩阵，采样间隔dt，值为1s
A = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]) 
# 过程噪声误差协方差矩阵，人为产生噪声
# Q = np.eye(4) * 1 
Q = np.eye(4)* 1
# 测量噪声协方差，人为产生测量噪声
R = np.diag([1,1,1,1]) *2
# 测量矩阵(对角)，默认初始值
H = np.array(np.eye(4))

# 预测数据设置
# 后验估计初始，人为增加噪声
x_po = x + randn(4).reshape(-1,1) 
# 初始后验误差协方差，设定初始值，误差协方差会自动更新
P_po = 100 * np.eye(4) 

# 存储位置结果使用
x_list = x
z_list = np.array([[],[],[],[]])
# 存储位置结果方差使用
x_po_list = x_po 
x_pr_list = np.array([[],[],[],[]])


# 卡尔曼滤波核心部分
# 仿真次数
iter_n = int(20)

for i in range(iter_n):
    # 真实值（预测的目标）
    # 生成带噪声的控制矩阵，一个多元正态（高斯）分布矩阵，设为一列，使用过程噪声Q
    w = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q , 1).reshape(-1,1)
    # 先验估计
    x = np.dot(A,x) + w
    # 存储估计结果
    x_list = np.hstack((x_list, x))
    
    # 观测值（测量得到的结果，一定存在误差，这个误差人为设定的，且比真是误差来的大 R ）
    # 生成观测误差矩阵，模拟观察状态
    v = np.random.multivariate_normal(np.zeros(R.shape[0]), R , 1).reshape(-1,1)
    # 对测量结果进行先验估计，人为添加噪音
    z = np.dot(H, x) + v
    # 存储测量结果（人为添加噪音）
    z_list = np.hstack((z_list,z))
    
    # 先验后验预测值（估计）
    # 预测下一个位置增量
    x_pr = np.dot(A, x_po)
    # 预测误差协方差矩阵，+Q 说明默认系统噪声协方差默认为对角矩阵，说明系统误差独立
    P_pr = np.dot(np.dot(A,P_po),A.T) + Q
    # 存储预测结果
    x_pr_list = np.hstack((x_pr_list,x_pr))
    
    # 计算卡尔曼增益
    # 计算卡尔曼增益分子部分
    k1 = np.dot(P_pr, H.T)
    # 计算卡尔曼增益分母部分
    k2 = np.dot(np.dot(H, P_pr), H.T) + R
    # 整合更新K值，K1与K2逆相乘
    K = np.dot(k1, np.linalg.inv(k2))
    
    # 修正估计值
    x_po = x_pr + np.dot(K,(z-np.dot(H,x_pr)))
    # 存储修正后的估计值
    x_po_list = np.hstack((x_po_list,x_po))
    # 更新误差协方差
    P_po = np.dot(np.eye(len(x)) - np.dot(K, H), P_pr)
    
# 绘图部分
# 图 1
plt.figure(figsize=(12, 12))
# 图例
legend_text = ['真实值', '测量值', '预测值', '最优值']
# 图像位置
plt.subplot(221)
# plt.ylim(140,300)
# x_axis表示时间轴
x_axis = range(0,iter_n+1)
# x_list 表示实际值，z_list 表示测量值，x_pr_list 表示预测值，x_po_list 表示后验估计值
plt.plot(x_axis, x_list[0, :],
         x_axis[1:], z_list[0, :],
         x_axis[1:], x_pr_list[0, :],
         x_axis, x_po_list[0, :], marker='.')
# 设置图标
plt.legend(legend_text)
plt.xlabel('Time')
plt.ylabel('X axis position')
# 图 2
plt.subplot(222)
# plt.ylim(300,340)
# 同图1，读取y轴位置变化情况
plt.plot(x_axis, x_list[1, :],
         x_axis[1:], z_list[1, :],
         x_axis[1:], x_pr_list[1, :],
         x_axis, x_po_list[1, :], marker='.')
plt.legend(legend_text)
plt.xlabel('Time')
plt.ylabel('Y axis position')
# 图3
plt.subplot(223)
# plt.ylim(10,20)
# 读取x轴速度数值变化情况
plt.plot(x_axis, x_list[2, :],
         x_axis[1:], z_list[2, :],
         x_axis[1:], x_pr_list[2, :],
         x_axis, x_po_list[2, :], marker='.')
plt.legend(legend_text)
plt.xlabel('Time')
plt.ylabel('X axis velocity')
# 图4
plt.subplot(224)
# plt.ylim(0,10)
x_axis = range(0,iter_n+1)
# 读取y轴速度情况
plt.plot(x_axis, x_list[3, :],
         x_axis[1:], z_list[3, :],
         x_axis[1:], x_pr_list[3, :],
         x_axis, x_po_list[3, :], marker='.')
plt.legend(legend_text)
plt.xlabel('Time')
plt.ylabel('Y axis velocity')
# 绘制图像
plt.show()

# 导入定义间隔函数
from matplotlib.ticker import MultipleLocator
fig = plt.figure(figsize=(6, 6))
# 在一张图内生成子图
ax = fig.add_subplot(1,1,1)
# 设定间隔为5
spacing = 5
minorLocator = MultipleLocator(spacing)
# 绘图，其中绘制X，Y位置，绘制X，Y速度
ax.plot(x_list[0,:], x_list[1, :],
    z_list[0,:], z_list[1, :],
    x_pr_list[0,:], x_pr_list[1, :],
    x_po_list[0,:], x_po_list[1, :], marker='.')
"""
    ax.plot(x_list[0,:], x_list[1, :],
    z_list[0,:], z_list[1, :], marker='.')
"""
ax.legend(legend_text)
ax.yaxis.set_minor_locator(minorLocator)
ax.xaxis.set_minor_locator(minorLocator)
ax.grid(which='both')
plt.xlabel('X axis Position')
plt.ylabel('Y axis Position')
# 绘图
plt.show()

# 绘制误差图像
plt.figure(figsize=(12, 12))
legend_text = ['最优估计误差', '测量误差', '预测误差']
error1 = np.abs(x_list-x_po_list) 
error2 = np.abs(x_list[:,1:]-z_list) 
error3 = np.abs(x_list[:,1:]-x_pr_list) 
# 计算误差方差
x_po_error = (error1[0,:]**2 + error1[1,:]**2 )**0.5
z_error = (error2[0,:]**2 + error2[1,:]**2 )**0.5
z_pr_error = (error3[0,:]**2 + error3[1,:]**2 )**0.5
# 绘制误差图
plt.plot(x_axis[1:],x_po_error[1:],
        x_axis[1:],z_error[:],
        x_axis[1:],z_pr_error[:])
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend(legend_text)
plt.show()

# plt.figure(figsize=(6, 6))