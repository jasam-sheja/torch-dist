#include <torch/extension.h>

#include <algorithm>
#include <numeric>
#include <functional>

#ifndef EUCLIDEAN_CPP
#define EUCLIDEAN_CPP
///////////////////////////////////////
/////////        cdist        /////////
///////////////////////////////////////
at::Tensor torch_cdist_org_forward(const at::Tensor &x1, const at::Tensor &x2)
{
    at::Tensor x1_norm = x1.pow(2).sum(-1, true);
    at::Tensor x1_pad = at::ones_like(x1_norm, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    at::Tensor x2_norm = x2.pow(2).sum(-1, true);
    at::Tensor x2_pad = at::ones_like(x2_norm, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    at::Tensor x1_ = at::cat({x1.mul(-2), x1_norm, x1_pad}, -1);
    at::Tensor x2_ = at::cat({x2, x2_pad, x2_norm}, -1);
    at::Tensor result = x1_.matmul(x2_.transpose(-2, -1));
    result.clamp_min_(0).sqrt_();
    return result;
}

at::Tensor eff_cdist_forward(const at::Tensor &x1, const at::Tensor &x2)
{
    at::Tensor x1_norm = x1.pow(2).sum(-1, true);
    at::Tensor x2_norm = x2.pow(2).sum(-1, true);
    at::Tensor sq_dist = at::baddbmm(
                             x2_norm.transpose(-2, -1),
                             x1,
                             x2.transpose(-2, -1),
                             1, -2)
                             .add_(x1_norm);
    return at::relu_(sq_dist).sqrt();
}

std::tuple<at::Tensor, at::Tensor> eff_cdist_backward(const at::Tensor &grad, const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &res)
{
    if (!grad.defined())
    {
        return std::tuple<at::Tensor, at::Tensor>(at::Tensor(), at::Tensor());
    }
    // handle case at 0 where we return a subgradient containing 0
    at::Tensor ratio = grad.div(res).nan_to_num_(0, 0, 0);
    //ratio.masked_fill_(res == 0, 0);
    return std::tuple<at::Tensor, at::Tensor>{
        x1.mul(ratio.sum(-1, true)).baddbmm_(ratio, x2, 1, -1),
        x2.mul(ratio.sum(-2, false).unsqueeze(-1)).baddbmm_(ratio.transpose(-2, -1), x1, 1, -1)};
}
at::Tensor eff_cdist_x1_backward(const at::Tensor &grad, const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &res)
{
    if (!grad.defined())
    {
        return at::Tensor();
    }
    // handle case at 0 where we return a subgradient containing 0
    at::Tensor ratio = grad.div(res).nan_to_num_(0, 0, 0);
    return x1.mul(ratio.sum(-1, true)).baddbmm_(ratio, x2, 1, -1);
}
at::Tensor eff_cdist_x2_backward(const at::Tensor &grad, const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &res)
{
    if (!grad.defined())
    {
        return at::Tensor();
    }
    // handle case at 0 where we return a subgradient containing 0
    at::Tensor ratio = grad.div(res).nan_to_num_(0, 0, 0);
    return x2.mul(ratio.sum(-2, false).unsqueeze(-1)).baddbmm_(ratio.transpose(-2, -1), x1, 1, -1);
}

/*
Only use with Cuda
*/

/*
on input x1=torch.Size([2, 1, 100]) and x2= torch.Size([2, 4, 100]) and some other
 like    x1=torch.Size([2, 1, 80]) and x2= torch.Size([2, 5, 80]) on the cpu its
not stable with this error
Intel MKL ERROR: Parameter 14 was incorrect on entry to cblas_sgemm_batch.
*/
at::Tensor eff2_cdist_forward(const at::Tensor &x1, const at::Tensor &x2)
{
    at::Tensor x1_norm = x1.pow(2).sum(-1, true);
    at::Tensor x2_norm = x2.pow(2).sum(-1, true);
    at::Tensor sq_dist = x1_norm.add(x2_norm.transpose(-2, -1)).baddbmm_(x1, x2.transpose(-2, -1), 1, -2);
    return at::relu_(sq_dist).sqrt();
}

std::tuple<at::Tensor, at::Tensor> eff_cdist_mem_backward(const at::Tensor &grad, const at::Tensor &x1, const at::Tensor &x2)
{
    if (!grad.defined())
    {
        return std::tuple<at::Tensor, at::Tensor>(at::Tensor(), at::Tensor());
    }
    // handle case at 0 where we return a subgradient containing 0
    at::Tensor res = eff2_cdist_forward(x1, x2);
    at::div_outf(grad, res, res);
    res.nan_to_num_(0, 0, 0);
    at::Tensor &ratio = res;
    //ratio.masked_fill_(res == 0, 0);
    return std::tuple<at::Tensor, at::Tensor>{
        x1.mul(ratio.sum(-1, true)).baddbmm_(ratio, x2, 1, -1),
        x2.mul(ratio.sum(-2, false).unsqueeze(-1)).baddbmm_(ratio.transpose(-2, -1), x1, 1, -1)};
}
at::Tensor eff_cdist_x1_mem_backward(const at::Tensor &grad, const at::Tensor &x1, const at::Tensor &x2)
{
    if (!grad.defined())
    {
        return at::Tensor();
    }
    // handle case at 0 where we return a subgradient containing 0
    at::Tensor res = eff2_cdist_forward(x1, x2);
    at::div_outf(grad, res, res);
    res.nan_to_num_(0, 0, 0);
    at::Tensor &ratio = res;
    return x1.mul(ratio.sum(-1, true)).baddbmm_(ratio, x2, 1, -1);
}
at::Tensor eff_cdist_x2_mem_backward(const at::Tensor &grad, const at::Tensor &x1, const at::Tensor &x2)
{
    if (!grad.defined())
    {
        return at::Tensor();
    }
    // handle case at 0 where we return a subgradient containing 0
    at::Tensor res = eff2_cdist_forward(x1, x2);
    at::div_outf(grad, res, res);
    res.nan_to_num_(0, 0, 0);
    at::Tensor &ratio = res;
    return x2.mul(ratio.sum(-2, false).unsqueeze(-1)).baddbmm_(ratio.transpose(-2, -1), x1, 1, -1);
}

///////////////////////////////////////
/////////        pdist        /////////
///////////////////////////////////////
at::Tensor eff_pdist_forward(const at::Tensor &x1)
{
    at::Tensor x1_norm = x1.pow(2).sum(-1, true);
    at::Tensor sq_dist = x1_norm.add(x1_norm.transpose(-2, -1)).baddbmm_(x1, x1.transpose(-2, -1), 1, -2);
    return at::relu_(sq_dist).sqrt();
}
at::Tensor eff_pdist_backward(const at::Tensor &grad, const at::Tensor &x, const at::Tensor &res)
{
    if (!grad.defined())
    {
        return at::Tensor();
    }
    // handle case at 0 where we return a subgradient containing 0
    at::Tensor ratio = grad.div(res).nan_to_num_(0, 0, 0);
    //ratio.masked_fill_(res == 0, 0);
    return x.mul(ratio.sum(-1, true)).baddbmm_(ratio, x, 1, -1).mul_(2);
}
at::Tensor eff_pdist_mem_backward(const at::Tensor &grad, const at::Tensor &x)
{
    if (!grad.defined())
    {
        return at::Tensor();
    }
    // handle case at 0 where we return a subgradient containing 0
    at::Tensor res = eff_pdist_forward(x);
    at::div_outf(grad, res, res);
    res.nan_to_num_(0, 0, 0);
    at::Tensor &ratio = res;
    //ratio.masked_fill_(res == 0, 0);
    return x.mul(ratio.sum(-1, true)).baddbmm_(ratio, x, 1, -1).mul_(2);
}


/////////////////////////////////////////////
/////////        cdistsquare        /////////
/////////////////////////////////////////////
at::Tensor eff_cdistsquare_forward(const at::Tensor &x1, const at::Tensor &x2)
{
    at::Tensor x1_norm = x1.pow(2).sum(-1, true);
    at::Tensor x2_norm = x2.pow(2).sum(-1, true);
    at::Tensor sq_dist = at::baddbmm(
                             x2_norm.transpose(-2, -1),
                             x1,
                             x2.transpose(-2, -1),
                             1, -2)
                             .add_(x1_norm);
    return at::relu_(sq_dist);
}

std::tuple<at::Tensor, at::Tensor> eff_cdistsquare_backward(const at::Tensor &grad, const at::Tensor &x1, const at::Tensor &x2)
{
    if (!grad.defined())
    {
        return std::tuple<at::Tensor, at::Tensor>(at::Tensor(), at::Tensor());
    }
    return std::tuple<at::Tensor, at::Tensor>{
        x1.mul(grad.sum(-1, true)).baddbmm_(grad, x2, 1, -1).mul_(2),
        x2.mul(grad.sum(-2, false).unsqueeze(-1)).baddbmm_(grad.transpose(-2, -1), x1, 1, -1).mul_(2)};
}
at::Tensor eff_cdistsquare_x1_backward(const at::Tensor &grad, const at::Tensor &x1, const at::Tensor &x2)
{
    if (!grad.defined())
    {
        return at::Tensor();
    }
    return x1.mul(grad.sum(-1, true)).baddbmm_(grad, x2, 1, -1).mul_(2);
}
at::Tensor eff_cdistsquare_x2_backward(const at::Tensor &grad, const at::Tensor &x1, const at::Tensor &x2)
{
    if (!grad.defined())
    {
        return at::Tensor();
    }
    return x2.mul(grad.sum(-2, false).unsqueeze(-1)).baddbmm_(grad.transpose(-2, -1), x1, 1, -1).mul_(2);
}

/////////////////////////////////////////////
/////////        pdistsquare        /////////
/////////////////////////////////////////////
at::Tensor eff_pdistsquare_forward(const at::Tensor &x)
{
    at::Tensor x_norm = x.pow(2).sum(-1, true);
    at::Tensor sq_dist = x_norm.add(x_norm.transpose(-2, -1)).baddbmm_(x, x.transpose(-2, -1), 1, -2);
    return at::relu_(sq_dist);
}

at::Tensor eff_pdistsquare_backward(const at::Tensor &grad, const at::Tensor &x)
{
    if (!grad.defined())
    {
        return at::Tensor();
    }
    return x.mul(grad.sum(-1, true)).baddbmm_(grad, x, 1, -1).mul_(4);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "C++ torch-dist";

    py::module euclidean = m.def_submodule("euclidean", "Euclidean distances module");
    euclidean.def("torch_cdist_org_forward", &torch_cdist_org_forward, "doc");

    euclidean.def("eff_cdist_forward", &eff_cdist_forward, "doc");
    euclidean.def("eff_cdist_backward", &eff_cdist_backward, "doc");
    euclidean.def("eff_cdist_x1_backward", &eff_cdist_x1_backward, "doc");
    euclidean.def("eff_cdist_x2_backward", &eff_cdist_x2_backward, "doc");

    euclidean.def("eff2_cdist_forward", &eff2_cdist_forward, "doc");

    euclidean.def("eff_cdist_mem_backward", &eff_cdist_mem_backward, "doc");
    euclidean.def("eff_cdist_x1_mem_backward", &eff_cdist_x1_mem_backward, "doc");
    euclidean.def("eff_cdist_x2_mem_backward", &eff_cdist_x2_mem_backward, "doc");
    
    euclidean.def("eff_pdist_forward", &eff_pdist_forward, "doc");
    euclidean.def("eff_pdist_backward", &eff_pdist_backward, "doc");
    euclidean.def("eff_pdist_mem_backward", &eff_pdist_mem_backward, "doc");
    
    euclidean.def("eff_cdistsquare_forward", &eff_cdistsquare_forward, "doc");
    euclidean.def("eff_cdistsquare_backward", &eff_cdistsquare_backward, "doc");
    euclidean.def("eff_cdistsquare_x1_backward", &eff_cdistsquare_x1_backward, "doc");
    euclidean.def("eff_cdistsquare_x2_backward", &eff_cdistsquare_x2_backward, "doc");
    
    euclidean.def("eff_pdistsquare_forward", &eff_pdistsquare_forward, "doc");
    euclidean.def("eff_pdistsquare_backward", &eff_pdistsquare_backward, "doc");
}

#endif