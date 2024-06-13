import streamlit as st


def main():
    st.title("URL Input and Display")

    # Get user input for multiple URLs
    urls_input = st.text_area("Enter URLs (one per line):", height=150)
    urls = urls_input.split("\n")  # Split into list


    urls = list(set(urls))
    lists = []
    print(urls)
    print(type(lists) == type(urls))

if __name__ == "__main__":
    main()
