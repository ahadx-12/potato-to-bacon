@equation(domain="classical")
def newton(F: Force(N), m: Mass(kg), a: Acceleration(m_per_s2)) -> Bool:
    return F == m*a
