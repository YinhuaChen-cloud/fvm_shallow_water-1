#include "sw.h"

//======================================================================
// 网格读取函数
//======================================================================
// 从STL文件中读取三维网格的几何信息，构建有限体积法所需的离散化结构
// 
// 主要功能：
// 1. 读取并去重顶点（vertices）
// 2. 创建计算单元（cells/控制体）
// 3. 识别并去重边界（edges）
// 4. 计算几何属性（面积、法向量、中心点等）
//
// 输入：STL格式的网格文件（三角形网格）
// 输出：填充sw结构体中的vertices, cells, edges等数据
//======================================================================

void read_grid(sw &sw)
{
    //======================================================================
    // 第一步：读取STL文件并去除重复顶点
    //======================================================================
    
    std::string file_line;              // 存储文件的每一行
    std::smatch file_line_match;        // 正则表达式匹配结果
    std::regex file_line_regex;         // 正则表达式对象

    // 正则表达式：匹配STL文件中的顶点行格式 "vertex x y z"
    file_line_regex.assign("\\s+vertex\\s(.+)\\s(.+)\\s(.+)");

    Eigen::Vector3d r1, r2, r3;         // 三角形的三个顶点坐标

    std::vector<int> vertices;          // 临时存储当前三角形的顶点索引
    Eigen::Vector3d vertex;             // 当前读取的顶点坐标
    Eigen::Vector3d face;               // 面（暂未使用）

    // 打开STL网格文件
    sw.file.open(sw.case_name + ".stl", std::fstream::in);

    int j = 0;
    if (sw.file.is_open())
    {
        // 逐行读取STL文件
        while (std::getline(sw.file, file_line))
        {
            file_line_match.empty();

            // 如果匹配到顶点行
            if (std::regex_search(file_line, file_line_match, file_line_regex))
            {
                // 提取x, y, z坐标
                vertex(0) = std::stod(file_line_match[1]);
                vertex(1) = std::stod(file_line_match[2]);
                vertex(2) = std::stod(file_line_match[3]);

                // 检查该顶点是否已存在（去重）
                bool vertex_exists = false;
                int vertex_i;
                for (int i = 0; i < sw.vertices.size(); i++)
                {
                    // 如果距离小于阈值1e-6，认为是同一顶点
                    // 矢量减法，取模(长度)
                    if ((sw.vertices[i] - vertex).norm() < 1e-6)
                    {
                        vertices.push_back(i);  // 使用已有顶点的索引
                        vertex_exists = true;
                        break;
                    }
                }

                // 如果是新顶点，添加到顶点列表
                if (!vertex_exists)
                {
                    sw.vertices.push_back(vertex);
                    vertices.push_back(sw.vertices.size() - 1);
                }

                // 每读取3个顶点，创建一个三角形单元
                if (vertices.size() == 3)
                {
                    cell cell(vertices[0], vertices[1], vertices[2]);
                    sw.cells.push_back(cell);
                    vertices.clear();  // 清空，准备读取下一个三角形
                }
            }
        }

        sw.file.close();

        sw.N_vertices = sw.vertices.size();
        sw.N_cells = sw.cells.size();

        //======================================================================
        // 第二步：识别并去除重复的边
        //======================================================================
        
        // 创建边矩阵（4行 x 3*N_cells列）
        // 每个三角形有3条边，每列代表一条边：
        // row 0: 边的第一个顶点索引
        // row 1: 边的第二个顶点索引
        // row 2: 左侧单元索引 (指的是在cells数组中的“左侧(较小)”索引)
        // row 3: 处理标记（0=未处理，1=已处理）
        Eigen::MatrixXd edges;
        edges = Eigen::MatrixXd::Zero(4, 3 * sw.N_cells);
        
        // 为每个单元的三条边填充边矩阵
        for (int i = 0; i < sw.N_cells; i++)
        {
            // 边1: vertex1 -> vertex2
            edges(0, 3 * (i + 1) - 3) = sw.cells[i].vertex1;
            edges(1, 3 * (i + 1) - 3) = sw.cells[i].vertex2;
            edges(2, 3 * (i + 1) - 3) = i;
            edges(3, 3 * (i + 1) - 3) = 0;

            // 边2: vertex2 -> vertex3
            edges(0, 3 * (i + 1) - 2) = sw.cells[i].vertex2;
            edges(1, 3 * (i + 1) - 2) = sw.cells[i].vertex3;
            edges(2, 3 * (i + 1) - 2) = i;
            edges(3, 3 * (i + 1) - 2) = 0;

            // 边3: vertex3 -> vertex1
            edges(0, 3 * (i + 1) - 1) = sw.cells[i].vertex3;
            edges(1, 3 * (i + 1) - 1) = sw.cells[i].vertex1;
            edges(2, 3 * (i + 1) - 1) = i;
            edges(3, 3 * (i + 1) - 1) = 0;
        }

        // 查找并合并重复的边
        // 两个单元共享的边，在矩阵中出现两次，方向相反
        for (int i = 0; i < 3 * sw.N_cells; i++)
        {
            if (edges(3, i) == 0)  // 如果该边尚未处理
            {
                int vertex1l, vertex2l, celll, cellr;

                vertex1l = edges(0, i);  // 左侧单元的边起点
                vertex2l = edges(1, i);  // 左侧单元的边终点

                celll = edges(2, i);     // 左侧单元索引
                cellr = -1;              // 右侧单元索引，初始化为-1（边界边）

                // 查找是否有另一条边是该边的反向（即共享边）
                for (int j = 0; j < 3 * sw.N_cells; j++)
                {
                    int vertex1r, vertex2r;

                    vertex1r = edges(0, j);  // 右侧单元的边起点
                    vertex2r = edges(1, j);  // 右侧单元的边终点

                    // 如果两条边的顶点顺序相反，说明是同一条边
                    if (vertex1l == vertex2r && vertex2l == vertex1r)
                    {
                        edges(3, j) = 1;      // 标记为已处理
                        cellr = edges(2, j);  // 记录右侧单元索引
                        break;
                    }
                }

                edges(3, i) = 1;  // 标记当前边为已处理

                // 创建边对象并添加到边列表
                // celll: 左侧单元，cellr: 右侧单元（-1表示边界）
                edge edge(vertex1l, vertex2l, celll, cellr);
                sw.edges.push_back(edge);
            }
        }

        sw.N_edges = sw.edges.size();

        //======================================================================
        // 第三步：计算单元的几何属性
        //======================================================================
        // 对每个三角形单元计算：
        // - 法向量 n（单位法向量）
        // - 面积 S
        // - 中心坐标 r（质心）
        
        for (int i = 0; i < sw.N_cells; i++)
        {
            Eigen::Vector3d r1, r2, r3, d1, d2, n;

            // 获取三角形的三个顶点坐标
            r1 = sw.vertices[sw.cells[i].vertex1];
            r2 = sw.vertices[sw.cells[i].vertex2];
            r3 = sw.vertices[sw.cells[i].vertex3];
            
            // 计算两条边向量
            d1 = r2 - r1;
            d2 = r3 - r1;
            
            // 通过叉积计算法向量（未归一化）
            n = d1.cross(d2);
            
            // 归一化法向量
            sw.cells[i].n = n / n.norm();
            
            // 计算三角形面积：|n|/2
            sw.cells[i].S = 0.5 * n.norm();
            
            // 计算三角形质心：(r1 + r2 + r3) / 3
            sw.cells[i].r = 1.0 / 3.0 * (r1 + r2 + r3);
        }

        //======================================================================
        // 第四步：计算边的几何属性
        //======================================================================
        // 对每条边计算：
        // - 法向量 n（单位法向量，指向外侧）
        // - 长度 l
        // - 中心坐标 r（中点）
        // - 识别边界边（cellr == -1）
        
        for (int i = 0; i < sw.N_edges; i++)
        {
            Eigen::Vector3d r1, r2, d, n;

            // 获取边的两个端点坐标
            r1 = sw.vertices[sw.edges[i].vertex1];
            r2 = sw.vertices[sw.edges[i].vertex2];
            
            // 边向量（从vertex1指向vertex2）
            d = r2 - r1;
            
            // 边法向量 = 边向量 × 左侧单元法向量
            // 这样计算的法向量指向左侧单元外部
            n = d.cross(sw.cells[sw.edges[i].celll].n);
            
            // 归一化法向量
            sw.edges[i].n = n / n.norm();
            
            // 边长度
            sw.edges[i].l = d.norm();
            
            // 边中点坐标
            sw.edges[i].r = 0.5 * (r1 + r2);

            // 如果边没有右侧单元（cellr == -1），说明是边界边
            // 将左侧单元标记为边界单元（type = 1）
            if (sw.edges[i].cellr == -1)
            {
                sw.cells[sw.edges[i].celll].type = 1;
            }
        }
    }
    else  // 文件打开失败
    {
        std::cout << "Failed reading grid file!" << std::endl;
        std::exit(-1);
    }
}
