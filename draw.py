import h5py
import matplotlib.pyplot as plt

def draw1():

    with h5py.File('../results/draw/fedavg.h5', 'r') as file:
    # 打开HDF5文件
        # 读取数据
        fed_acc = file['rs_test_acc']  # 假设Y轴数据集在文件中被命名为'y_dataset'

        # 解压缩数据
        fed_acc = fed_acc[:].flatten()
    with h5py.File('../results/draw/uxishu.h5', 'r') as file:
    # 打开HDF5文件
        # 读取数据
        uxishu = file['rs_test_acc']  # 假设Y轴数据集在文件中被命名为'y_dataset'

        # 解压缩数据
        uxishu = uxishu[:].flatten()
    with h5py.File('../results/draw/abxishu.h5', 'r') as file:
    # 打开HDF5文件
        # 读取数据
        abxishu = file['rs_test_acc']  # 假设Y轴数据集在文件中被命名为'y_dataset'

        # 解压缩数据
        abxishu = abxishu[:].flatten()


    data_x = range(501)  # 假设X轴数据集在文件中被命名为'x_dataset'
    # 绘制折线图
    plt.plot(data_x, fed_acc, label='FedAvg')
    plt.plot(data_x, abxishu, label='Heterogeneous Mask')
    plt.plot(data_x, uxishu, label='Common Mask')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy of target(%)')
    plt.title('Cifar10-pathological')
    plt.legend(loc='lower right')

    # 显示图形
    plt.show()

def draw2():

    rate = f'20'
    with h5py.File('../results/acc/patch' + rate + '.h5', 'r') as file:
        # 打开HDF5文件
        # 读取数据
        patch = file['rs_test_acc']  # 假设Y轴数据集在文件中被命名为'y_dataset'

        # 解压缩数据
        patch_acc = patch[:].flatten()
    with h5py.File('../results/acc/xishu' + rate + '.h5', 'r') as file:
        # 打开HDF5文件
        # 读取数据
        xishu = file['rs_test_acc']  # 假设Y轴数据集在文件中被命名为'y_dataset'

        # 解压缩数据
        xishu_acc = xishu[:].flatten()
    # with h5py.File('../results/acc/abxishu.h5', 'r') as file:
    #     # 打开HDF5文件
    #     # 读取数据
    #     abxishu = file['rs_test_acc']  # 假设Y轴数据集在文件中被命名为'y_dataset'
    #
    #     # 解压缩数据
    #     abxishu = abxishu[:].flatten()

    data_x = range(501)  # 假设X轴数据集在文件中被命名为'x_dataset'
    # 绘制折线图
    plt.plot(data_x, patch_acc, label='patch')
    # plt.plot(data_x, abxishu, label='Heterogeneous Mask')
    plt.plot(data_x, xishu_acc, label='xishu')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy of target(%)')
    plt.title('Cifar10-pathological-rate' + rate)
    plt.legend(loc='lower right')

    # 显示图形
    plt.show()


def draw3():
    import matplotlib.pyplot as plt
    import numpy as np

    # 生成示例数据
    x = np.linspace(0, 1, 9)
    mse_value = [0.393965793, 0.394017267, 0.39326579, 0.39395935, 0.392935491, 0.587089539, 0.548713467, 0.373074168, 0.302665004]
    mse_upper = [0.396988243, 0.398471057, 0.396333516, 0.3971816, 0.398513019, 1.361232519, 1.39640069, 0.469521284, 0.393379688]
    mse_lower = [0.389612526, 0.391868234, 0.389886945, 0.389094085, 0.389904737, 0.391014308, 0.163838968, 0.216387957, 0.197167367]

    LPIPS_value = [0.783823788, 0.784542704, 0.784773946, 0.784678113, 0.78531841, 0.718232632, 0.65627023, 0.645857912, 0.587682229]
    LPIPS_upper = [0.799066186, 0.80068326, 0.794827402, 0.791140318, 0.802466631, 0.781768441, 0.794716656, 0.781762719, 0.784781933]
    LPIPS_lower = [0.749553204, 0.771538675, 0.780642509, 0.778174698, 0.7724002, 0.454738766, 0.412469387, 0.483167499, 0.315938264]

    psnr_value = [4.045497322, 4.044934559, 4.053210926, 4.045590687, 4.056914711, 2.97214632, 3.705271149, 4.418874359, 5.357476044]
    psnr_upper = [4.093670845, 4.068599224, 4.090613365, 4.099453926, 4.090415001, 4.078073978, 7.855828285, 6.647669792, 7.05164957]
    psnr_lower = [4.012223244, 3.996032238, 4.019392014, 4.010108948, 3.995574951, 2.69864234, 3.583447266, 3.283447266, 4.051880836]

    mser_value = [0.145099482, 0.145724908, 0.145240203, 0.145332971, 0.145464373, 0.12374119, 0.105227912, 0.07838814, 0.078089081]
    mser_upper = [0.145737976, 0.146555185, 0.145585045, 0.146111518, 0.145869106, 0.14650552, 0.145932436, 0.145978361, 0.14538157]
    mser_lower = [0.144287109, 0.14487505, 0.144371554, 0.144943327, 0.145042732, 0.037493922, 0.002875042, 0.023156758, 0.008782102]


    # 绘制折线图和填充上下区间范围
    plt.plot(x, mse_value, color='blue', label='MSE')
    plt.fill_between(x, mse_upper, mse_lower, color='skyblue', alpha=0.5)
    plt.plot(x, LPIPS_value, color='red', label='LPIPS')
    plt.fill_between(x, LPIPS_upper, LPIPS_lower, color='salmon', alpha=0.5)
    plt.plot(x, psnr_value, color='blue', label='PSNR')
    plt.fill_between(x, psnr_upper, psnr_lower, color='skyblue', alpha=0.5)
    plt.plot(x, mser_value, color='red', label='MSE-R')
    plt.fill_between(x, mser_upper, mser_lower, color='salmon', alpha=0.5)

    # 设置图例
    plt.legend()

    # 添加标题和标签
    plt.title('Multiple Lines Plot with Upper and Lower Ranges')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 显示图形
    plt.show()

if __name__ == "__main__":
    draw3()
    # KSGeneral()