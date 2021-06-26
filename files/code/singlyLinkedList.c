#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct node {
    char name[200];
    int age;
    struct node* next;
} node;

node *head, *tail;

node* create_node(const char* name, int age);
void push_back(const char* name, int age);
void push_front(const char* name, int age);
void insert_after(const char *name, int age, const char *key);
void delete_head(void);
void delete_tail(void);
void delete_node(const char* name);
void delete_list(void);
void print_list(void);

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

node* create_node(const char* name, int age) {
    // allocate memory of size 'node';
    node* student = (node*) malloc(sizeof(node));
    // create a new node based on the given arguments;
    strcpy(student->name, name);
    student->age = age;

    return student;
}

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
        // free the curr since its unused;
        free(student);
    } else { // if key is somewhere in the middle of the list;
        // append the rest of the list after the newly created node;
        student->next = curr->next;  
        // connect the new node after the key; 
        curr->next = student;
    }
}

void delete_head(void) {
    node* curr = head;
    // set head's 'next' as the new head, then free head;
    head = head->next;
    free(curr);
}

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

void delete_node(const char* name) {
    node* curr = head;
    // traverse to the node to be deleted;
    while (strcmp(curr->next->name, name) != 0 && curr != NULL) {
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