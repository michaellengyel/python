def main():
    ### LISTS ###
    print("LISTS: Is a collection which is ordered and changeable. Allows duplicate members.")

    m_list = []
    m_list.append(1)
    m_list.append("test")
    m_list.append(3.42453)
    m_list.append(None)

    print(m_list)
    print(m_list[2])
    print(m_list.count(3.42453))

    # List comprehension
    squares = [x ** 2 for x in range(10)]
    print(squares)

    ### TUPLES ###
    print("TUPLES: Is a collection which is ordered and unchangeable. Allows duplicate members.")

    m_tuple_1 = (2, 1, 4)
    m_tuple_2 = (m_tuple_1, "word", 42)
    m_tuple_3 = (m_tuple_2, 1, 4.4)

    print(m_tuple_1)
    print(m_tuple_2)
    print(m_tuple_3)

    print(m_tuple_1[2])

    ### SETS ###
    print("SETS: Is a collection which is unordered and unindexed. No duplicate members.")

    m_set_1 = {"red", "green", "blue"}
    m_set_2 = set(("red", "green", "blue"))
    # Use set() function to create a set. An empty {}
    print(m_set_1)
    print(m_set_2)

    m_set_1.add("red")
    m_set_1.add("purple")

    print(m_set_1)

    ### DICTIONARY ###
    print("DICTIONARY: Is a collection which is ordered and changeable. No duplicate members.")

    m_dict_1 = {1: "red", 2: "green", 3: "blue"}
    m_dict_2 = {"red": [1, 2, 3], "green": 3.2332, 3: "apple"}

    print(m_dict_1)
    print(m_dict_1[2])
    print(len(m_dict_1))

    print(m_dict_2)
    print(m_dict_2["green"])
    m_dict_2["gin"] = 1827
    print(m_dict_2)


if __name__ == '__main__':
    main()
