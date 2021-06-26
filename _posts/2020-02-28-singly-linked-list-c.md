---
title: Singly Linked List in C
date: 2020-02-28
permalink: /posts/2020/02/singly-linked-list-c/
tags:
  - Data Structures
---

According to [Wikipedia](https://en.wikipedia.org/wiki/Linked_list), a **linked list** is a linear collection of data elements, whose order is not given by their physical placement in memory. Instead, each element points to the next. It is a data structure consisting of a collection of nodes which together represent a sequence.

We'll implement a Singly Linked List using the C language. The complete code for this post can be found [here](https://github.com/w11wo/w11wo.github.io/blob/master/files/code/singlyLinkedList.c).

The following code is **based** on a lecture by Rhio Sutoyo, S.Kom., M.Sc. in **Data Structures** course.

## Header Files

The only header files we'll be using are the following

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
```

## Node Struct

A `node` is just a single element inside the list, which in this case represents a student's information with their `name` and `age`. Also, it has a `pointer` to the next `node`.

```c
typedef struct node {
    char name[200];
    int age;
    struct node* next;
} node;
```

Notice that we also use `typedef` which allows us to omit the `struct` keyword in the instantiation of a `node`.

## Function Prototypes

Since we are writing in C, we need to first prototype every function we're going to implement below our `main` function. Here is the list of functions we'll be implementing

```c
node* create_node(const char* name, int age);
void push_back(const char* name, int age);
void push_front(const char* name, int age);
void insert_after(const char *name, int age, const char *key);
void delete_head(void);
void delete_tail(void);
void delete_node(const char* name);
void delete_list(void);
void print_list(void);
```

## Global Head and Tail

For this example, we will create a global variable called `head` and `tail`, which denotes the first element and the last element in the list respectively.

```c
node *head, *tail;
```

## Creating a Node

To create our student `node`, we will implement the following function.

```c
node* create_node(const char* name, int age) {
    // allocate memory of size 'node';
    node* student = (node*) malloc(sizeof(node));
    // create a new node based on the given arguments;
    strcpy(student->name, name);
    student->age = age;

    return student;
}
```

## Inserting a Node into the Linked List

### Push Back

There are multiple ways to insert a `node` to a linked list, and first we'll be implementing a function called `push_back` which will append a new `node` to the end of the list.

```c
void push_back(const char* name, int age) {
    // create a new node;
    node* student = create_node(name, age);

    if (head == NULL) { // if list is empty;
        head = student;
        tail = student;
        head->next = NULL;
        tail->next = NULL;
    } else {
        // add new node to tail;
        tail->next = student;
        // set new node as tail;
        tail = student;
        tail->next = NULL;
    }
}
```

### Push Front

Then, we can also implement a function called `push_front` to add a `node` to the front of the list.

```c
void push_front(const char* name, int age) {
    // create a new node;
    node* student = create_node(name, age);

    if (head == NULL) { // if list is empty;
        head = student;
        tail = student;
        head->next = NULL;
        tail->next = NULL;
    } else {
        // set the existing list as 'next' of the newly created node;
        student->next = head;
        // set the newly created node as head;
        head = student;
    }
}
```

### Insert After

Lastly, a function called `insert_after` will allow us to insert a new `node` after a particular `key` `node`.

```c
void insert_after(const char* name, int age, const char* key) {
    // create a new node;
    node* student = create_node(name, age);

    node* curr = head;
    // traverse to the node which the new node will be placed after;
    while (curr != NULL) {
        if (strcmp(curr->name, key) == 0) {
            break;
        } else {
            curr = curr->next;
        }
    }

    if (curr == NULL) { // if key is not found;
        printf("\"%s\" is not in the list.\n", key);
    } else if (curr == tail) { // if key is tail;
        // append to tail;
        push_back(name, age);
        // free student since it's unused;
        free(student);
    } else { // if key is somewhere in the middle of the list;
        // append the rest of the list after the newly created node;
        student->next = curr->next;
        // connect the new node after the key;
        curr->next = student;
    }
}
```

## Deleting a Node from the Linked List

### Delete Head

We can delete the `head` of the list.

```c
void delete_head(void) {
    node* curr = head;
    // set head's 'next' as the new head, then free head;
    head = head->next;
    free(curr);
}
```

### Delete Tail

Likewise the `tail` of the list.

```c
void delete_tail(void) {
    node* curr = head;
    // traverse to second to the last node in the list;
    while (curr->next->next != NULL) {
        curr = curr->next;
    }
    // set the second to the last node as the new tail;
    tail = curr;
    // free the old tail;
    free(curr->next);
    tail->next = NULL;
}
```

### Delete a Particular Node based on name

We can delete a particular `node` based on its `name`.

```c
void delete_node(const char* name) {
    node* curr = head;
    // traverse to the node before the node to be deleted;
    while (curr != NULL && strcmp(curr->next->name, name) != 0) {
        curr = curr->next;
    }

    if (curr == NULL) { // if 'name' is not in the list;
        printf("\"%s\" is not in the list.\n", name);
    } else if (curr->next == tail) { // if 'name' is tail;
        delete_tail();
    } else if (curr->next == head) { // if 'name' is head;
        delete_head();
    } else { // if 'name' is somewhere in the middle of the list;
        // set tmp as the node to be deleted;
        node* tmp = curr->next;
        // skip the node to be deleted;
        curr->next = tmp->next;
        // delete/free node;
        free(tmp);
    }
}
```

## Delete Linked List

Allow for the deletion of the entire list.

```c
void delete_list(void) {
    if (head == NULL) { // if list is empty;
        printf("List is empty.\n");
    } else {
        while (head != NULL) {
            // set the node after head to be the new head;
            node* curr = head;
            head = head->next;
            // free the old head;
            free(curr);
        }
        // reset head and tail;
        head = NULL;
        tail = NULL;
    }
}
```

## Print Linked List

For convenience, create a function to `print` the entire list.

```c
void print_list(void) {
    if (head == NULL) { // if list is empty;
        printf("List is empty.\n");
    } else {
        node* curr = head;
        // traverse through each node;
        while (curr != NULL) {
            printf("Name: %s, Age: %d\n", curr->name, curr->age);
            curr = curr->next;
        }
    }
}
```

## Main Function

Lastly, we'll demonstrate how the `main` function looks like.

```c
int main(void) {

    push_back("Steven", 19); // [Steven]
    push_front("Bill", 24); // [Bill, Steven]

    delete_head(); // [Steven]

    push_back("John", 14); // [Steven, John]

    delete_tail(); // [Steven]

    push_front("Anne", 16); // [Anne, Steven]

    insert_after("Greg", 26, "Anne"); // [Anne, Greg, Steven]

    delete_node("Greg"); // [Anne, Steven]

    print_list();

    delete_list(); // []

    return 0;

}
```

## Conclusion

Linked List allows for dynamic list allocation, relatively more efficient in the insertion and deletion of an element, but requires linear access time.

Linked List will also be useful in the implementation of other data structures such as `stack` and `queue` which will be discussed in future posts.
