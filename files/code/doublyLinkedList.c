#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct node {
    char name[200];
    double gpa;
    struct node* next;
    struct node* prev;
} node;

node *head, *tail;

node* create_node(const char* name, double gpa);
void sorted_push(const char* name, double gpa);
void delete_node(const char* key);
void print_list(void);
void print_reversed_list(void);

int main(void) {

    sorted_push("Steven", 3.5); // [Steven]
    sorted_push("Bill", 2.0); // [Bill, Steven]
    sorted_push("John", 3.7); // [Bill, Steven, John]
    sorted_push("Ace", 2.5); // [Bill, Ace, Steven, John]

    delete_node("Ace"); // [Bill, Steven, John]
    delete_node("B");
    print_list();

    return 0;

}

node* create_node(const char* name, double gpa) {
    // allocate memory of size 'node';
    node* student = (node*) malloc(sizeof(node));
    // create a new node based on the given arguments;
    strcpy(student->name, name);
    student->gpa = gpa;

    return student;
}

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