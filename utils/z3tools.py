from z3 import *

def get_variables(solver):
    """从 Solver 对象中提取所有未解释的常量（用户定义的变量）"""
    variables = set()
    for assertion in solver.assertions():
        # 递归遍历表达式树，收集变量
        stack = [assertion]
        while stack:
            expr = stack.pop()
            if is_const(expr):
                # 检查是否为未解释的常量（即用户定义的变量）
                if expr.decl().kind() == Z3_OP_UNINTERPRETED:
                    variables.add(expr)
            else:
                # 将子表达式加入栈中继续遍历
                stack.extend(expr.children())
    return variables