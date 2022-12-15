---
title: Hash Tables, Collisions, and Separate Chaining
date: 2020-04-11
permalink: /posts/2020/04/doubly-linked-list-c/
tags:
  - Data Structures
---

According to [Wikipedia](https://en.wikipedia.org/wiki/Hash_table), a **hash table** or sometimes called **hash map** is a a data structure that implements an associative array abstract data type, a structure that can map keys to values.

Unlike Linked Lists, Hash Tables allow for an O(1) time complexity when searching, which is a powerful tool knowing that a Linked List requires O(n) complexity.
However, there may be possible collisions when inserting a new data whose key has already been used. This causes the search time complexity to have a O(n) worst case, just like a Linked List.

We'll implement a Hash Table using the C language.
For each key in the hash table, we'll implement it using a `node` of a Singly Linked List to cater for separate chaining.

The complete code for this post can be found [here](https://github.com/w11wo/wilsonwongso.dev/blob/master/files/code/hashTable.c).

## Header Files

The only header files we'll be using are the following

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
```

## Hash Table Size

To keep the size of the hash table constant, we will define the maximum number of keys using the `define` keyword. There will be only 26 keys which is based on the 26 alphabets.

```c
#define MAX_N 26
```

## Node Struct

Each node in the linked list only consists a `name` `string` and a `pointer` `next` which points to the next node in the linked list.

```c
typedef struct node {
    char name[200];
    struct node* next;
} node;
```

## Function Prototypes

Since we are writing in C, we need to first prototype every function we're going to implement below our `main` function. Here is the list of functions we'll be implementing

```c
node* create_node(const char* name);
int hash(const char* name);
void insert(node* root[], const char* name);
char* search(node* root[], const char* name);
void print_list(node* head, int idx);
void print_table(node* root[]);
```

## Creating a Node

To create our `node`, we will implement the following function. It is almost identical to the one in the Singly Linked List [blog post](https://wilsonwongso.dev/blog/code/c/2020/02/28/singly-linked-list-c.html).

```c
node* create_node(const char* name) {
    node* student = (node*) malloc(sizeof(node));

    strcpy(student->name, name);
    student->next = NULL;

    return student;
}
```

## Hashing Function

The hashing function we'll be using is called the division method.
Specifically, we take the first letter of the string, convert it to its lower case letter, and return its ASCII equivalent.

```c
int hash(const char* name) {
    return tolower(name[0]) - 'a';
}
```

With that we will get only 26 possible keys for a string, which is the same as the `MAX_N` we've defined earlier.
There are various ways to create a better hashing function with aims to reduce collisions, which we'll discuss later in the blog.

## Inserting a Node into the Hash Table

First we utilize `create_node()` to allocate the memory required and create the `student` `node`.
Then, with the hashing function we can get the corresponding key for the given `string` `name`.

Inserting the first node of a key is as simple as getting the address of the corresponding `head` and setting it to be the newly created `student`.

However, we will need to address possible collisions and we do this by a method called separate chaining, which basically means appending the next `node` with the same key to the linked list.

```c
void insert(node* root[], const char* name) {
    node* student = create_node(name);

    int key = hash(name);

    node** head = &root[key];

    if (*head == NULL) { // if the head of a particular key is still NULL;
        *head = student;
    } else { // separate chaining, i.e. push the new node to the back of the linked list.
        node* curr = *head;
        while(curr->next != NULL) {
            curr = curr->next;
        }
        curr->next = student;
    }
}
```

## Searching for a name inside the Hash Table.

Searching is one of the most powerful features of a Hash Table as discussed previously.
To implement searching by `name`, we will be using linear search since we are using linked lists.

Just like insertion, we cater to the possible scenarios like a `NULL` `head` and traversing through a linked list otherwise.

```c
char* search(node* root[], const char* name) {
    int key = hash(name);

    node* head = root[key];

    if (head == NULL) {
        return NULL;
    } else {
        node* curr = head;
        while(curr != NULL) {
            if (strcmp(curr->name, name) == 0) {
                return curr->name;
            }
            curr = curr->next;
        }
        return NULL;
    }
}
```

## Printing a Linked List

Since each key in the hash table uses a linked list, we need to prepare a function which prints a linked list.
We would also like to print the index of the linked list in the hash table, so we pass another parameter called `idx`.

```c
void print_list(node* head, int idx) {
    if (head == NULL) {
        return;
    } else {
        printf("[%d] ", idx);
        node* curr = head;
        while (curr != NULL) {
            printf("%s", curr->name);
            curr = curr->next;
            if (curr != NULL) {
                printf(" -> ");
            }
        }
        printf("\n");
    }
}
```

## Printing the Hash Table

Finally, we can implement a function to print every non-empty linked lists and its contents.

```c
void print_table(node* root[]) {
    for (int i = 0; i < MAX_N; ++i) {
        print_list(root[i], i);
    }
}
```

## Main Function

We'll demonstrate how the `main` function looks like.

```c
int main(void) {

    node* root[MAX_N] = {NULL};

    insert(root, "Apple");
    insert(root, "Orange");
    insert(root, "Papaya");
    insert(root, "Avocado");
    insert(root, "Blueberry");
    insert(root, "Peach");
    insert(root, "Plum");

    char* find_banana = search(root, "Banana");
    if (find_banana != NULL) {
        printf("%s found\n", find_banana);
    }

    char* find_avocado = search(root, "Avocado");
    if (find_avocado != NULL) {
        printf("%s found\n", find_avocado);
    }

    print_table(root);

    return 0;

}
```

The output looks something like this

```
Avocado found
[0] Apple -> Avocado
[1] Blueberry
[14] Orange
[15] Papaya -> Peach -> Plum
```

## A Better Hash Table and Hashing Function

As said previously, the previously implemented function is not the best when trying to reduce collisions. We would like to avoid collisions as much as possible to maintain O(1) search time.

A reference code was given by my lecturer as an example of a good hashing function and hash table, as well as explanations as to why certain decisions were made.

**Credit belongs to the author of this code**.

There are several aspects to improve a hash table, namely its size and the hashing function being used.

### Size of Hash Table

Firstly, the size of the hash table plays a role in the distribution of the key in the hash table.

A good size would be a prime number, since it has very few factors. While non-prime numbers cause distribution of keys to be not uniformly distributed.

Simply put, a non-uniform distribution of keys causes other keys which are not factors of the size of the hash table to be of high probability in being empty.

For example, if our choice was to use 12 as the size of the hash table.
The key 3, a factor of 12, along with its multiples (0, 3, 6, 9, ...) will be more likely to be filled while others empty, thus increasing the chance of collision.

### Hash Function

Aside from being fast to be computed, a good hashing function distributes keys as uniformly possible.

To do so, we sum the ASCII equivalents of every character in the string to make the key as unique as possible.

We also add a so-called zero-padding if ever an empty string is allowed to prevent it affecting universality.

In addition, every time we sum the ASCII, we add a base number, which is strictly greater than the number of different values of each individual letters. Doing so further increases the range of the possible keys hence reducing collisions.

For example, since there are 26 possible lowercase letters, a base number like 31 is preferable. The base number 31 is also used by a method called `hashCode()` in Java's `String` [class](<https://docs.oracle.com/javase/6/docs/api/java/lang/String.html#hashCode()>).

A more detailed explanation can be found in this Wikipedia page about [Universal hashing](https://en.wikipedia.org/wiki/Universal_hashing#Hashing_strings).

### Code Implementation

With the said changes, our hash table starts off to be of size 97.

```c
int hashTable[97];
```

While the hash function looks like

```c
int hash(const char *str) {
    int len = strlen(str);

    int base = 31;

    int MODPRIME = 97;

    int ret = 0;

    for (int i = 0; i < len; i++) {
        ret = (ret * base) + (str[i] - 'a' + 1);
        ret = ret % MODPRIME;
    }

    return (ret * base) % MODPRIME;
}
```

## Conclusion

It is very interesting to see how small details of a hash table can greatly affect its performance and the math behind it.
With the capabilities of a hash table, searching can be greatly improved in comparison to the previously discussed data structures.

To read up more on hash tables, the reference code also linked to very resourceful [notes](http://cseweb.ucsd.edu/~kube/cls/100/Lectures/lec16/lec16-2.html#pgfId-982677) from UC San Diego.
