//======================================================================
// 浅水方程有限体积法求解器
// 用于求解二维浅水方程组，支持多种数值格式：
// - 显式/隐式时间推进
// - 不同的Riemann解法器
// - 不同阶数的时间积分格式
//======================================================================

#include <iostream>
#include <string>
#include <chrono>

#include "sw.h" 

int main(int argc, char *argv[])
{
    // 检查命令行参数，需要提供算例配置文件
	if (argc < 2)
	{
		std::cout << "Usage: sw_solver <case>" << std::endl;
		std::exit(-1);
	}

	//======================================================================
	// 变量定义
	//======================================================================
	
	sw sw;                                      // 浅水方程求解器结构体
	Eigen::VectorXd Q;                          // 解向量(h,hu,hv)
	Eigen::VectorXd Q_tmp;                      // 临时解向量(用于RK积分)
	Eigen::VectorXd k1, k2, k3, k4;             // RK积分的中间变量
	Eigen::VectorXd R;                          // 残差向量
	Eigen::MatrixXd A;                          // 隐式求解的系统矩阵
	Eigen::BiCGSTAB<Eigen::MatrixXd> solver;    // BiCGSTAB迭代求解器(隐式方法)
	double t;                                   // 当前物理时间

	sw.case_name = argv[1];
	solver.setTolerance(1e-6);
	solver.setMaxIterations(20);
	t = 0.0;

	// Config
	////////////////////////////////////////////////////////////////////

	std::cout << "Reading config...";

	read_config(sw);

	std::cout << "Done!" << std::endl;

	std::cout << std::endl;
	std::cout << "dt = " << sw.dt << std::endl;
	std::cout << "N_timesteps = " << sw.N_timesteps << std::endl;
	std::cout << "output_frequency = " << sw.output_frequency << std::endl;
	std::cout << "riemann_solver = " << sw.riemann_solver << std::endl;
	std::cout << "integrator = " << sw.integrator << std::endl;
	std::cout << "implicit = " << sw.implicit << std::endl;
	std::cout << std::endl;

	// Grid
	////////////////////////////////////////////////////////////////////

	std::cout << "Reading grid...";

	read_grid(sw);

	std::cout << "Done!" << std::endl;

	std::cout << "N_vertices=" << sw.N_vertices << std::endl;
	std::cout << "N_cells=" << sw.N_cells << std::endl;
	std::cout << "N_edges=" << sw.N_edges << std::endl;

	//======================================================================
	// 初始条件设置
	//======================================================================
	
	// 初始化解向量，每个单元有3个变量(h,hu,hv)
	Q = Eigen::VectorXd::Zero(3 * sw.N_cells);

	// 设置初始条件：模拟溃坝问题
	// x <= -0.3: h = 1.5, u = v = 0 (高水位)
	// x > -0.3:  h = 1.0, u = v = 0 (低水位)
	for (int i = 0; i < sw.N_cells; i++)
	{
		if (sw.cells[i].r[0] <= -0.3)  // 左侧高水位区域
		{
			sw.cells[i].Q(0) = 1.5;     // 水深h = 1.5
			sw.cells[i].Q(1) = 0.0;     // 动量hu = 0
			sw.cells[i].Q(2) = 0.0;     // 动量hv = 0
		}
		else                            // 右侧低水位区域
		{
			sw.cells[i].Q(0) = 1.0;     // 水深h = 1.0
			sw.cells[i].Q(1) = 0.0;     // 动量hu = 0
			sw.cells[i].Q(2) = 0.0;     // 动量hv = 0
		}

		Q(3 * i) = sw.cells[i].Q(0);
		Q(3 * i + 1) = sw.cells[i].Q(1);
		Q(3 * i + 2) = sw.cells[i].Q(2);
	}

	// Jacobian (if implicit=1)
	////////////////////////////////////////////////////////////////////

	if (sw.implicit == 1)
	{
		std::cout << "Evaluating jacobian..." << std::endl;

		Eigen::VectorXd dR;

		A = Eigen::MatrixXd::Zero(3 * sw.N_cells, 3 * sw.N_cells);
		R = residual(sw, Q);

		for (int i = 0; i < 3 * sw.N_cells; i++)
		{
			Q_tmp = Q;
			Q_tmp(i) += 1e-6;
			dR = residual(sw, Q_tmp);
			for (int j = 0; j < 3 * sw.N_cells; j++)
			{
				A(j, i) = -(dR(j) - R(j)) / (1e-6);
			}
			A(i, i) += 1.0 / sw.dt;
		}

		solver.compute(A);
	}

	//======================================================================
	// 时间推进主循环
	//======================================================================

	std::cout << "Performing " << sw.N_timesteps << " steps..." << std::endl;

	for (int n = 0; n < sw.N_timesteps; n++)
	{
		std::cout << "Step " << n << ", t=" << t << std::endl;

		// 计算当前时间步的残差
		// 残差 = -Δt * (数值通量 - 源项)
		R = residual(sw, Q);

		// 数值稳定性检查：检查解是否包含NaN
		// 如果出现NaN，说明计算发散，需要终止程序
		for (int i = 0; i < sw.N_cells; i++)
		{
			if (std::isnan(Q(i)))
			{
				std::cout << "Residual is nan, stopping..." << std::endl;
				std::exit(-1);
			}
		}

		// 时间推进方法选择
		if (sw.implicit == 1)  // 隐式方法
		{
			// 使用BiCGSTAB求解线性方程组 A*dQ = R
			Eigen::VectorXd dQ;
			dQ = solver.solve(R);

			// 输出求解信息
			std::cout << "Iterations: " << solver.iterations() << std::endl;
			std::cout << "Error: " << solver.error() << std::endl;

			// 更新解向量
			Q += dQ;
		}
		else  // 显式方法
		{
			if (sw.integrator == 0)  // 一阶显式欧拉方法
			{
				Q += sw.dt * R;  // Q^(n+1) = Q^n + dt*R^n
			}
			if (sw.integrator == 1)  // 二阶Runge-Kutta方法(RK2)
			{
				k1 = sw.dt * R;                          // k1 = dt*R(Q^n)
				Q_tmp = Q + k1 / 2.0;                    // Q* = Q^n + k1/2
				k2 = sw.dt * residual(sw, Q_tmp);        // k2 = dt*R(Q*)
				Q += k2;                                 // Q^(n+1) = Q^n + k2
			}
			if (sw.integrator == 2)  // 四阶Runge-Kutta方法(RK4)
			{
				k1 = sw.dt * R;                          // k1 = dt*R(Q^n)
				Q_tmp = Q + k1 / 2.0;                    // Q1 = Q^n + k1/2
				k2 = sw.dt * residual(sw, Q_tmp);        // k2 = dt*R(Q1)
				Q_tmp = Q + k2 / 2.0;                    // Q2 = Q^n + k2/2
				k3 = sw.dt * residual(sw, Q_tmp);        // k3 = dt*R(Q2)
				Q_tmp = Q + k3;                          // Q3 = Q^n + k3
				k4 = sw.dt * residual(sw, Q_tmp);        // k4 = dt*R(Q3)
				// Q^(n+1) = Q^n + (k1 + 2k2 + 2k3 + k4)/6
				Q += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
			}
		}

		// 更新物理时间
		t += sw.dt;

		// 结果输出
		// output_frequency控制输出频率:
		// - output_frequency = 0: 不输出
		// - output_frequency > 0: 每隔output_frequency步输出一次
		if (sw.output_frequency > 0)
		{
			if (n % sw.output_frequency == 0)
			{
				write_results(sw, n);  // 将当前时间步的结果写入文件
			}
		}
	}

	std::cout << "Done!" << std::endl;

	return 0;
}
