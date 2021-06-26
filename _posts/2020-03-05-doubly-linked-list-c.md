---
title: Doubly Linked List in C
date: 2020-03-05
permalink: /posts/2020/03/doubly-linked-list-c/
tags:
  - Data Structures
---

After learning how to implement Singly Linked List, we're going to implement Doubly Linked List, which is similar to Singly Linked List, but with the addition of a `prev` `pointer` which points to the node before it.

We'll implement a Doubly Linked List using the C language. The complete code for this post can be found [here](https://github.com/w11wo/w11wo.github.io/blob/master/files/code/doublyLinkedList.c).

The following code is **based** on a lecture by Rhio Sutoyo, S.Kom., M.Sc. in **Data Structures** course.

## Header Files

The only header files we'll be using are the following

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
```

## Node Struct

A `node` is just a single element inside the list, which in this case represents a student's information with their `name` and `gpa`. Also, it has a `pointer` to the next and previous `node`.

```c
typedef struct node {
    char name[200];
    double gpa;
    struct node* next;
    struct node* prev;
} node;
```

Notice that we also use `typedef` which allows us to omit the `struct` keyword in the instantiation of a `node`.

## Function Prototypes

Since we are writing in C, we need to first prototype every function we're going to implement below our `main` function. Here is the list of functions we'll be implementing

```c
node* create_node(const char* name, double gpa);
void sorted_push(const char* name, double gpa);
void delete_node(const char* key);
void print_list(void);
void print_reversed_list(void);
```

## Global Head and Tail

For this example, we will create a global variable called `head` and `tail`, which denotes the first element and the last element in the list respectively.

```c
node *head, *tail;
```

## Creating a Node

To create our student `node`, we will implement the following function.

```c
node* create_node(const char* name, double gpa) {
    // allocate memory of size 'node';
    node* student = (node*) malloc(sizeof(node));
    // create a new node based on the given arguments;
    strcpy(student->name, name);
    student->gpa = gpa;

    return student;
}
```

## Sorted Push

Instead of implementing push front, back, or middle, we're going to create a function which will automatically insert a node in ascending order of `gpa`.

```c
void sorted_push(const char* name, double gpa) {
    // create a new node;
    node* student = create_node(name, gpa);

    if (head == NULL) { // if list is empty;
        head = student;
        tail = student;
        head->next = NULL;
        tail->next = NULL;
        head->prev = NULL;
        tail->prev = NULL;
    } else {
        node* curr = head;
        // traverse to the node with gpa greater than the one being pushed;
        while (curr != NULL && curr->gpa < student->gpa) {
            curr = curr->next;
        }

        if (curr == head) { // if the head already has a value greater than the new node's;
            // append old head to the new node;
            student->next = head;
            head->prev = student;
            // set new node as new head;
            head = student;
            head->prev = NULL;
        } else if (curr == NULL) { // if we've reached the node after tail, i.e. all values are less than the value being pushed;
            // append new node to tail;
            tail->next = student;
            student->prev = tail;
            // set new node as new tail;
            tail = student;
            tail->next = NULL;
            free(curr);
        } else { // if we have to push the new node in the middle;
            // connect the current's previous node to the new node;
            curr->prev->next = student;
            student->prev = curr->prev;
            // connect curr as the next of the new node;
            student->next = curr;
            curr->prev = student;
        }
    }
}
```

## Delete a Node Based on Name

We can delete a particular `node` based on its `name`.

```c
void delete_node(const char* key) {
    if (head == NULL) { // if list is empty;
        printf("List is empty.\n");
    } else {
        node* curr = head;
        // traverse to the node to be deleted;
        while (curr != NULL && strcmp(curr->name, key) != 0) {
            curr = curr->next;
        }

        if (curr == NULL) { // if key is not in the list;
            printf("\"%s\" is not in the list.\n", key);
        } else if (curr == head && curr == tail) { // if key the only node in the list;
            // delete node;
            free(curr);
            // reset head and tail;
            head = NULL;
            tail = NULL;
        } else if (curr == head) { // if key is head;
            // set old head's next as new head;
            head = head->next;
            head->prev = NULL;
            // free old head since its no longer used;
            free(curr);
        } else if (curr == tail) { // if key is tail;
            // set the node before old tail to be the new tail;
            tail = tail->prev;
            tail->next = NULL;
            // fre old tail;
            free(curr);
        } else {
            // skip the node being deleted;
            curr->prev->next = curr->next;
            curr->next->prev = curr->prev;
            // free the deleted node;
            free(curr);
        }
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
            printf("Name: %-10s GPA: %.2lf\n", curr->name, curr->gpa);
            curr = curr->next;
        }
    }
}
```

## Print Linked List in Reverse Order

Since we have the `prev` `pointer`, we can easily `print` the list in reverse order.

```c
void print_reversed_list(void) {
    if (head == NULL) { // if list is empty;
        printf("List is empty.\n");
    } else {
        node* curr = tail;
        // traverse through each node backwards;
        while (curr != NULL) {
            printf("Name: %-10s GPA: %.2lf\n", curr->name, curr->gpa);
            curr = curr->prev;
        }
    }
}
```

## Main Function

Lastly, we'll demonstrate how the `main` function looks like.

```c
int main(void) {

    sorted_push("Steven", 3.5); // [Steven]
    sorted_push("Bill", 2.0); // [Bill, Steven]
    sorted_push("John", 3.7); // [Bill, Steven, John]
    sorted_push("Ace", 2.5); // [Bill, Ace, Steven, John]

    delete_node("Ace"); // [Bill, Steven, John]

    print_list();

    return 0;

}
```

## Conclusion

With doubly linked list, we can easily move forward and backward from a node, which will highly ease the process of adding a node, printing in reverse order, and others which singly linked list would have a difficulty of doing.
