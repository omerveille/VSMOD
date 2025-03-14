import unittest
from ransac_slicer.branch_tree import BranchTree, BranchTreeItem, TreeColumnRole
import qt


class Branch_treeTest(unittest.TestCase):
    def test_01_BranchTreeItem___init__(self):
        item = BranchTreeItem("node")

        self.assertEqual(item.nodeId, "node")
        self.assertEqual(item.text(0), "node")
        self.assertIsNotNone(item.icon(TreeColumnRole.VISIBILITY_CENTER))
        self.assertIsNotNone(item.icon(TreeColumnRole.VISIBILITY_CONTOUR))
        self.assertIsNotNone(item.icon(TreeColumnRole.DELETE))

    def test_02_BranchTreeItem_updateText(self):
        item = BranchTreeItem("node")
        item.nodeId = "new_node_id"
        item.updateText()

        self.assertEqual(item.text(0), "new_node_id")

    def test_03_BranchTree___init__(self):
        tree = BranchTree()

        self.assertEqual(tree.columnCount, 4)
        self.assertEqual(tree.headerItem().text(0), "Branch Name")
        self.assertEqual(tree.headerItem().text(1), " Center")
        self.assertEqual(tree.headerItem().text(2), " Contour")
        self.assertEqual(tree.headerItem().text(3), "")

    def test_04_BranchTree_clear(self):
        tree = BranchTree()
        tree._branchDict = {"node": BranchTreeItem("node")}
        tree.addTopLevelItem(tree._branchDict["node"])
        tree.clear()

        self.assertEqual(len(list(tree.getNodeList())), 0)

    def test_05_BranchTree_setItemSelected(self):
        tree = BranchTree()
        item = BranchTreeItem("node")
        tree.addTopLevelItem(item)
        tree.setItemSelected(item)
        selectedItems = tree.selectedItems()

        self.assertIn(item, selectedItems)

    def test_06_BranchTree_isInTree(self):
        tree = BranchTree()
        tree._branchDict["node"] = BranchTreeItem("node")

        self.assertTrue(tree.isInTree("node"))
        self.assertFalse(tree.isInTree("nonexistent"))

    def test_07_BranchTree_isRoot(self):
        tree = BranchTree()
        item = BranchTreeItem("node")
        tree._branchDict["node"] = item
        tree.addTopLevelItem(item)
        self.assertTrue(tree.isRoot("node"))
        childId = "child"
        childItem = BranchTreeItem(childId)
        item.addChild(childItem)
        tree._branchDict[childId] = childItem

        self.assertFalse(tree.isRoot(childId))

    def test_08_BranchTree_keyPressEvent(self):
        tree = BranchTree()

        class DummyKeyEvent:
            def key(self):
                return qt.Qt.Key_Delete

        item = BranchTreeItem("node")
        tree.addTopLevelItem(item)
        tree.setCurrentItem(item)
        keyPressedFlag = {"key": None}

        def on_key_pressed(item_param, key):
            keyPressedFlag["key"] = key

        tree.keyPressed.connect(on_key_pressed)
        tree.keyPressEvent(DummyKeyEvent())

        self.assertEqual(keyPressedFlag["key"], qt.Qt.Key_Delete)

    def test_09_BranchTree_onHeaderClicked(self):
        tree = BranchTree()
        headerClickedFlag = {"column": None}

        def on_header_clicked(column):
            headerClickedFlag["column"] = column

        tree.headerClicked.connect(on_header_clicked)
        tree.onHeaderClicked(2)

        self.assertEqual(headerClickedFlag["column"], 2)

    def test_10_BranchTree__takeItem(self):
        tree = BranchTree()
        item = tree._takeItem("node")
        self.assertEqual(item.nodeId, "node")
        tree._branchDict["node"] = item
        taken_item = tree._takeItem("node")

        self.assertEqual(taken_item, item)

    def test_11_BranchTree__removeFromParent(self):
        tree = BranchTree()
        parent = BranchTreeItem("parent")
        child = BranchTreeItem("child")
        parent.addChild(child)
        tree.addTopLevelItem(parent)

        tree._removeFromParent(child)
        self.assertNotIn(child, [parent.child(i) for i in range(parent.childCount())])

    def test_12_BranchTree__insertNode(self):
        tree = BranchTree()
        item = tree._insertNode("node", None)

        self.assertEqual(item.nodeId, "node")
        self.assertTrue(tree.isInTree("node"))

        parent_item = BranchTreeItem("parent_node")
        tree._branchDict["parent_node"] = parent_item
        tree.addTopLevelItem(parent_item)
        child_item = tree._insertNode("child_node", "parent_node")

        self.assertEqual(child_item.nodeId, "child_node")
        self.assertEqual(tree.getParentNodeId("child_node"), "parent_node")

    def test_13_BranchTree_insertAfterNode(self):
        tree = BranchTree()
        tree.insertAfterNode("node", None)

        self.assertTrue(tree.isInTree("node"))
        tree.insertAfterNode("child13", "node")

        self.assertEqual(tree.getParentNodeId("child13"), "node")

    def test_14_BranchTree_removeNode(self):
        tree = BranchTree()
        parent_item = BranchTreeItem("parent")
        child_item = BranchTreeItem("child")
        parent_item.addChild(child_item)
        tree._branchDict["parent"] = parent_item
        tree._branchDict["child"] = child_item
        tree.addTopLevelItem(parent_item)
        result = tree.removeNode("child")

        self.assertTrue(result)
        self.assertNotIn("child", tree._branchDict)

    def test_15_BranchTree__removeIntermediateItem(self):
        tree = BranchTree()
        parent_item = BranchTreeItem("parent")
        intermediate_item = BranchTreeItem("intermediate")
        child_item = BranchTreeItem("child")
        parent_item.addChild(intermediate_item)
        intermediate_item.addChild(child_item)
        tree._branchDict["parent"] = parent_item
        tree._branchDict["intermediate"] = intermediate_item
        tree._branchDict["child"] = child_item
        tree._removeIntermediateItem(intermediate_item, "intermediate")
        children_ids = [
            parent_item.child(i).nodeId for i in range(parent_item.childCount())
        ]

        self.assertIn("child", children_ids)
        self.assertNotIn("intermediate", tree._branchDict)

    def test_16_BranchTree_getParentNodeId(self):
        tree = BranchTree()
        parent_item = BranchTreeItem("parent")
        child_item = BranchTreeItem("child")
        parent_item.addChild(child_item)
        tree._branchDict["parent"] = parent_item
        tree._branchDict["child"] = child_item

        self.assertEqual(tree.getParentNodeId("child"), "parent")

    def test_17_BranchTree_getChildrenNodeId(self):
        tree = BranchTree()
        parent_item = BranchTreeItem("parent")
        child_item1 = BranchTreeItem("child_1")
        child_item2 = BranchTreeItem("child_2")
        parent_item.addChild(child_item1)
        parent_item.addChild(child_item2)
        tree._branchDict["parent"] = parent_item
        tree._branchDict["child_1"] = child_item1
        tree._branchDict["child_2"] = child_item2
        children = tree.getChildrenNodeId("parent")

        self.assertIn("child_1", children)
        self.assertIn("child_2", children)

    def test_18_BranchTree_getNodeList(self):
        tree = BranchTree()
        tree._branchDict = {
            "node_1": BranchTreeItem("node_1"),
            "node_2": BranchTreeItem("node_2"),
        }
        node_list = list(tree.getNodeList())

        self.assertIn("node_1", node_list)
        self.assertIn("node_2", node_list)

    def test_19_BranchTree_getTreeWidgetItem(self):
        tree = BranchTree()
        item = BranchTreeItem("node")
        tree._branchDict["node"] = item

        self.assertEqual(tree.getTreeWidgetItem("node"), item)
        self.assertIsNone(tree.getTreeWidgetItem("nonexistent"))

    def test_20_BranchTree_getText(self):
        tree = BranchTree()
        item = BranchTreeItem("node")
        tree._branchDict["node"] = item
        tree.addTopLevelItem(item)

        self.assertEqual(tree.getText("node"), "node")
        self.assertEqual(tree.getText("nonexistent"), "")

    def test_21_BranchTree_isLeaf(self):
        tree = BranchTree()
        item = BranchTreeItem("node")
        tree._branchDict["node"] = item
        tree.addTopLevelItem(item)

        self.assertTrue(tree.isLeaf("node"))
        child_item = BranchTreeItem("child")
        item.addChild(child_item)
        tree._branchDict["child"] = child_item

        self.assertFalse(tree.isLeaf("node"))

    def test_22_BranchTree_enforceOneRoot(self):
        tree = BranchTree()
        root_1 = BranchTreeItem("root_1")
        root_2 = BranchTreeItem("root_2")
        tree._branchDict["root_1"] = root_1
        tree._branchDict["root_2"] = root_2
        tree.addTopLevelItem(root_1)
        tree.addTopLevelItem(root_2)
        tree.enforceOneRoot()

        self.assertEqual(tree.topLevelItemCount, 1)

        top_item = tree.topLevelItem(0)
        self.assertGreater(top_item.childCount(), 0)
