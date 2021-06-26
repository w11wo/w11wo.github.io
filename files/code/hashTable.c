#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_N 26

typedef struct node {
    char name[200];
    struct node* next;
} node;

node* create_node(const char* name);
int hash(const char* name);
void insert(node* root[], const char* name);
char* search(node* root[], const char* name);
void print_list(node* head, int idx);
void print_table(node* root[]);

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

node* create_node(const char* name) {
    node* student = (node*) malloc(sizeof(node));

    strcpy(student->name, name);
    student->next = NULL;

    return student;
}

int hash(const char* name) {
    return tolower(name[0]) - 'a';
}

void insert(node* root[], const char* name) {
    node* student = create_node(name);

    int key = hash(name);

    node** head = &root[key];

    if (*head == NULL){
        *head = student;
    } else {
        node* curr = *head;
        while(curr->next != NULL){
            curr = curr->next;
        }
        curr->next = student;
    }
}

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

void print_table(node* root[]) {
    for (int i = 0; i < MAX_N; ++i) {
        print_list(root[i], i);
    }
}