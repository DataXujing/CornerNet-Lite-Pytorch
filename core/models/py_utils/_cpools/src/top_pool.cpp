#include <torch/torch.h>

#include <vector>

std::vector<at::Tensor> top_pool_forward(
    at::Tensor input
) {
    // Initialize output
    at::Tensor output = at::zeros_like(input);

    // Get height
    int64_t height = input.size(2);

    output.copy_(input);

    //这边ind可能是index吧
    for (int64_t ind = 1; ind < height; ind <<= 1) {
        // 默认步长为1，这里没写出来，参数是output用作常数的输出，然后是dim维度是2,起点0,终点高度-ind
        at::Tensor max_temp = at::slice(output, 2, 0, height-ind);
        at::Tensor cur_temp = at::slice(output, 2, 0, height-ind);
        at::Tensor next_temp = at::slice(output, 2, ind, height);
        // 用maxout激活单元输出其中的最大值
        at::max_out(max_temp, cur_temp, next_temp);
    }

    return { 
        output
    };
}

std::vector<at::Tensor> top_pool_backward(
    at::Tensor input,
    at::Tensor grad_output
) {
    // auto声明变量时根据初始化表达式自动推断该变量的类型、声明函数时函数返回值的占位符
    auto output = at::zeros_like(input);

    int32_t batch   = input.size(0);
    int32_t channel = input.size(1);
    int32_t height  = input.size(2);
    int32_t width   = input.size(3);

    auto max_val = torch::zeros({batch, channel, width}, at::device(at::kCUDA).dtype(at::kFloat));
    auto max_ind = torch::zeros({batch, channel, width}, at::device(at::kCUDA).dtype(at::kLong));

    // select 维度2的在height-1处的那一个，我们看得到上面高度的维度在2上，就取最高的那个
    auto input_temp = input.select(2, height - 1);
    // 不管怎么样，先默认它是最大的
    max_val.copy_(input_temp);

    // 然后用高度填充最大数的位置矩阵
    max_ind.fill_(height - 1);

    // 现在output矩阵还是0呢，同样选取高度切片
    auto output_temp      = output.select(2, height - 1);
    // grad_output说我也要切片哒
    auto grad_output_temp = grad_output.select(2, height - 1);
    // 是我jojo哒
    output_temp.copy_(grad_output_temp);

    // 前面的最大数维度重新拓展 高度的维度
    auto un_max_ind = max_ind.unsqueeze(2);
    // 你妈的 为什么又要了两个掩码层
    auto gt_mask    = torch::zeros({batch, channel, width}, at::device(at::kCUDA).dtype(at::kByte));
    auto max_temp   = torch::zeros({batch, channel, width}, at::device(at::kCUDA).dtype(at::kFloat));
    // 我们又重新进入高度的循环
    for (int32_t ind = 1; ind < height; ++ind) {
        // 中间取高度为H-index的那个 去切片
        input_temp = input.select(2, height - ind - 1);
        at::gt_out(gt_mask, input_temp, max_val);

        at::masked_select_out(max_temp, input_temp, gt_mask);
        max_val.masked_scatter_(gt_mask, max_temp);
        max_ind.masked_fill_(gt_mask, height - ind - 1);

        grad_output_temp = grad_output.select(2, height - ind - 1).unsqueeze(2);
        output.scatter_add_(2, un_max_ind, grad_output_temp);
    }

    return {
        output
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward", &top_pool_forward, "Top Pool Forward",
        py::call_guard<py::gil_scoped_release>()
    );
    m.def(
        "backward", &top_pool_backward, "Top Pool Backward",
        py::call_guard<py::gil_scoped_release>()
    );
}
