while True:
    try:
        print('Doing something else')
    except KeyboardInterrupt:
        print('All done')
        # If you actually want the program to exit
        break

print("Finally..")