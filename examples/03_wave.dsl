@equation(domain="classical")
def wave(u, x: Length(m), t: Time(s), c: Speed(m_per_s)) -> Bool:
    return d2(u,t) == c**2 * d2(u,x)
