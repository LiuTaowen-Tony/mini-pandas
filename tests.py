from data_frame import DataFrame, Series


df_dict = {
    "SKU": ["X4E", "T3D", "F8D", "C7X"],
    "price": [7.0, 3.5, 8.0, 5.0],
    "sales": [5, 3, 1, 10],
    "taxed": [False, False, True, False]
}

df = DataFrame.from_dict(df_dict)

df_dict2 = {
    "SKU": ["X4E", "T3D", "F8D", "C7X"],
    "id": [1, 2, 3, 4],
}

df2 = DataFrame.from_dict(df_dict2)

df.left_join(df2, "SKU")

df_dict2 = {
    "SKU": ["X4E", "C7X", "XXX", "F8D", ],
    "id": [1, 2, 3],
}

position_map = [1, 4, 3]

# key_to_pos : left_key -> int_position



# ---------

def test(fn):
    fn()
    print(f"{fn.__name__} \n-------------------\n")

@test
def array_does_not_have_none_is_not_optional():
    l = [1.2, 2.3, 4.5]
    series = Series.from_array_like(l)
    print(series)

@test
def reject_incompatible_types_when_constructing():
    l = [1.2, 2.3, 4.5, "hello"]
    try:
        series = Series.from_array_like(l,)
    except ValueError as e:
        print(e)

@test
def can_perform_numeric_comparison():
    price_greater_than_4 = df["price"] > 4
    print(price_greater_than_4)

@test
def can_perform_reverse_comparison():
    price_greater_than_4 = 4 < df["price"] 
    print(price_greater_than_4)

@test
def reject_incomptible_type_computation():
    f = [1.2, 2.3, 4.5]
    fs = Series.from_array_like(f)
    b = [True, False, True]
    bs = Series.from_array_like(b)

    try:
        fs + bs
    except ValueError as e:
        print(e)

    try:
        fs ^ bs
    except ValueError as e:
        print(e)

    try:
        False & fs
    except ValueError as e:
        print(e)


@test
def reject_incompatible_length():
    l = [1.2, 2.3, 4.5]
    series = Series.from_array_like(l)
    s = [1, 2]
    series2 = Series.from_array_like(s)
    try:
        series + series2
    except ValueError as e:
        print(e)

@test
def can_perform_masking():
    print(df["price"][df["taxed"]])

@test
def can_build_series_has_none():
    all_none = [None, None, None, None]
    all_none_series = Series.from_array_like(all_none)
    print(all_none_series)
    l = [1.2, None, 2.3, 4.5]
    none_series = Series.from_array_like(l)
    print(none_series)

@test
def can_perform_arith_on_series_has_none():
    l = [1.2, None, 2.3, 4.5]
    none_series = Series.from_array_like(l)
    print(none_series)
    print(none_series + df["price"])

@test
def can_perform_boolean_operations():
    print(~df["taxed"])
    print(df)
    print(df["taxed"] ^ df["taxed"])
    print(df["taxed"] ^ False)

@test
def can_select_series_by_name():
    print(df["price"])

@test
def can_select_by_mask():
    print(df[df["price"] > 4])


@test
def integration_test():
    term1 = df["price"] + 5.0 > 10.0
    term2 = df["sales"] > 3
    term3 = ~df["taxed"]
    print(term1)
    print(term2)
    print(term3)
    mask = (df["price"] + 5.0 > 10.0) & (df["sales"] > 3) & ~df["taxed"]
    print(mask)
    result = df[(df["price"] + 5.0 > 10.0) & (df["sales"] > 3) & ~df["taxed"]]["SKU"]
    print(result)

@test
def large_scale_test():
    import random
    import string
    import time
    import pandas as pd

    def random_string(length):
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

    n = 100000
    df_dict = {
        "SKU": [random_string(3) for _ in range(n)],
        "price": [random.uniform(0, 10) for _ in range(n)],
        "sales": [random.randint(0, 100) for _ in range(n)],
        "taxed": [random.choice([True, False]) for _ in range(n)]
    }

    start = time.time()
    df = DataFrame.from_dict(df_dict)
    print(f"Time to construct DataFrame: {time.time() - start}")

    start = time.time()
    mask = (df["price"] + 5.0 > 10.0) & (df["sales"] > 3) & ~df["taxed"]
    print(f"Time to construct mask: {time.time() - start}")

    start = time.time()
    result = df[mask]["SKU"]
    print(f"Time to filter DataFrame: {time.time() - start}")

    start = time.time()
    pdf = pd.DataFrame(df_dict)
    mask = (pdf["price"] + 5.0 > 10.0) & (pdf["sales"] > 3) & ~pdf["taxed"]
    result = pdf[mask]["SKU"]
    print(f"Time to filter Pandas DataFrame: {time.time() - start}")
